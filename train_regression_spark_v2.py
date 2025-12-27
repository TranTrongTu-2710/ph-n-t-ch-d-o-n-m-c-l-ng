# train_regression_spark_v2.py
"""
Train mô hình dự báo lương (Spark MLlib) - Phiên bản nâng cấp.

MỤC TIÊU:
- Đọc dữ liệu Gold (đã có vector features).
- Chia tập Train/Test (80/20).
- Huấn luyện 2 mô hình: Linear Regression và GBT Regressor.
- Đánh giá sai số (RMSE, MAE).
- Lưu mô hình tốt nhất.
"""

import os
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# ===== CẤU HÌNH =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data2")
FEATURES_PATH = os.path.join(DATA_DIR, "salary_features_spark.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def create_spark():
    spark = (
        SparkSession.builder
        .appName("Train_Regression_Spark_v2_Advanced")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def evaluate_real_scale(pred_df):
    """
    Hàm đánh giá sai số trên thang đo thực tế (USD).
    Vì mô hình dự đoán log(lương), nên cần expm1() để đổi về lương thật.
    """
    df2 = (
        pred_df
        .withColumn("salary_true", F.expm1(F.col("label")))      # Lương thật = exp(log_lương) - 1
        .withColumn("salary_pred", F.expm1(F.col("prediction"))) # Lương dự đoán = exp(log_dự_đoán) - 1
        .filter(F.col("salary_pred").isNotNull())
    )
    # MAE: Sai số tuyệt đối trung bình (Trung bình lệch bao nhiêu USD)
    mae = df2.select(F.avg(F.abs(F.col("salary_true") - F.col("salary_pred"))).alias("mae")).collect()[0]["mae"]
    # RMSE: Căn bậc hai của sai số bình phương trung bình (Phạt nặng các sai số lớn)
    rmse = df2.select(F.sqrt(F.avg((F.col("salary_true") - F.col("salary_pred"))**2)).alias("rmse")).collect()[0]["rmse"]
    return float(mae), float(rmse)

def main():
    spark = create_spark()

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing: {FEATURES_PATH}")

    print(f"[INFO] Dang doc features tu: {FEATURES_PATH}")
    df = spark.read.parquet(FEATURES_PATH)

    # Lọc dữ liệu hợp lệ
    df = df.filter(F.col("ConvertedCompYearly").isNotNull() & (F.col("ConvertedCompYearly") > 0))

    # Outlier filter (1% - 99%): Lọc bỏ 1% lương thấp nhất và cao nhất để mô hình ổn định hơn
    q1, q99 = df.approxQuantile("ConvertedCompYearly", [0.01, 0.99], 0.01)
    df = df.filter((F.col("ConvertedCompYearly") >= F.lit(q1)) & (F.col("ConvertedCompYearly") <= F.lit(q99)))

    # Chuyển label sang log scale (log1p)
    # Lý do: Lương thường phân phối lệch phải (Right-skewed), log giúp nó phân phối chuẩn hơn -> Tốt cho Linear Regression
    df = (df
          .withColumn("label", F.log1p(F.col("ConvertedCompYearly")))
          .select("features", "label")
          .cache())
    
    print("Rows after outlier filter:", df.count())
    print(f"Quantile range (Salary): {q1:.2f} to {q99:.2f}")

    # Chia tập Train (80%) và Test (20%)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_df, test_df = train_df.cache(), test_df.cache()
    print(f"Train: {train_df.count()} | Test: {test_df.count()}")

    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # =========================
    # MODEL 1: Linear Regression
    # =========================
    print("\n[INFO] Training LinearRegression...")
    # ElasticNet: Kết hợp L1 (Lasso) và L2 (Ridge) regularization để tránh overfitting
    lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.01, elasticNetParam=0.5)
    lr_model = lr.fit(train_df)
    
    pred_lr = lr_model.transform(test_df)
    rmse_lr = evaluator_rmse.evaluate(pred_lr)
    r2_lr = evaluator_r2.evaluate(pred_lr)
    mae_real_lr, rmse_real_lr = evaluate_real_scale(pred_lr)

    print(f"[LR] RMSE(log): {rmse_lr:.4f} | R2(log): {r2_lr:.4f}")
    print(f"[LR] MAE(real): {mae_real_lr:,.2f} | RMSE(real): {rmse_real_lr:,.2f}")
    
    # Lưu model LR
    lr_model.write().overwrite().save(os.path.join(MODEL_DIR, "lr_model"))

    # =========================
    # MODEL 2: GBT Regressor (Gradient Boosted Trees)
    # =========================
    print("\n[INFO] Training GBTRegressor (Tuned)...")
    # GBT thường tốt hơn LR vì bắt được các mối quan hệ phi tuyến tính (Non-linear)
    # maxDepth=8: Cây sâu hơn để học các pattern phức tạp
    gbt = GBTRegressor(
        featuresCol="features", 
        labelCol="label", 
        maxIter=80, 
        maxDepth=8, 
        stepSize=0.1, 
        subsamplingRate=0.8,
        seed=42
    )
    gbt_model = gbt.fit(train_df)
    
    pred_gbt = gbt_model.transform(test_df)
    rmse_gbt = evaluator_rmse.evaluate(pred_gbt)
    r2_gbt = evaluator_r2.evaluate(pred_gbt)
    mae_real_gbt, rmse_real_gbt = evaluate_real_scale(pred_gbt)

    print(f"[GBT] RMSE(log): {rmse_gbt:.4f} | R2(log): {r2_gbt:.4f}")
    print(f"[GBT] MAE(real): {mae_real_gbt:,.2f} | RMSE(real): {rmse_real_gbt:,.2f}")

    # Lưu model GBT
    gbt_model.write().overwrite().save(os.path.join(MODEL_DIR, "gbt_model"))

    # So sánh và kết luận
    best = ("LR", rmse_lr) if rmse_lr <= rmse_gbt else ("GBT", rmse_gbt)
    print(f"\n[RESULT] Best model by RMSE(log): {best[0]} (RMSE={best[1]:.4f})")

    spark.stop()

if __name__ == "__main__":
    main()
