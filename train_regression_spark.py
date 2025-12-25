# train_regression_spark.py
"""
Train mô hình dự báo lương (Spark MLlib) từ output step_3_gold_features.py

Input:
  - data2/salary_features_spark.parquet
    schema chính: features (vector), ConvertedCompYearly (double)

Output:
  - models/lr_model
  - models/gbt_model

Đánh giá:
  - RMSE/R2 trên log1p(salary)
  - MAE/RMSE trên salary thật (expm1)
"""

import os
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# ===== PATH ổn định theo vị trí file =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data2")

FEATURES_PATH = os.path.join(DATA_DIR, "salary_features_spark.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def create_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("Train_Regression_Spark_Light")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "32")  # nhẹ hơn cho máy local
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def evaluate_real_scale(pred_df):
    """
    label = log1p(salary)
    prediction = log1p(salary_pred)
    => salary_pred = expm1(prediction)
    """
    df2 = (
        pred_df
        .withColumn("salary_true", F.expm1(F.col("label")))
        .withColumn("salary_pred", F.expm1(F.col("prediction")))
        .filter(F.col("salary_pred").isNotNull())
    )

    mae = df2.select(F.avg(F.abs(F.col("salary_true") - F.col("salary_pred"))).alias("mae")).collect()[0]["mae"]
    rmse = df2.select(
        F.sqrt(F.avg((F.col("salary_true") - F.col("salary_pred")) ** 2)).alias("rmse")
    ).collect()[0]["rmse"]

    return float(mae), float(rmse)


def main():
    spark = create_spark()

    # 1) Load features
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"❌ Không tìm thấy: {FEATURES_PATH}\n"
            f"➡️ Hãy chạy step_3_gold_features.py trước để tạo file features."
        )

    print(f"📥 Đang đọc features từ: {FEATURES_PATH}")
    df = spark.read.parquet(FEATURES_PATH)

    required_cols = {"features", "ConvertedCompYearly"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"❌ Input thiếu cột {required_cols}. Cột hiện có: {df.columns}")

    # 2) Tạo label log1p + lọc sạch
    df = (
        df
        .filter(F.col("ConvertedCompYearly").isNotNull() & (F.col("ConvertedCompYearly") > 0))
        .withColumn("label", F.log1p(F.col("ConvertedCompYearly")))
        .select("features", "label")
        .cache()
    )
    total = df.count()
    print("✅ Total rows (usable):", total)

    # 3) Split train/test
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_df = train_df.cache()
    test_df = test_df.cache()
    print("Train rows:", train_df.count())
    print("Test rows :", test_df.count())

    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # =========================
    # MODEL 1: Linear Regression (nhẹ)
    # =========================
    print("\n🚀 Training LinearRegression...")
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        regParam=0.05,
        elasticNetParam=0.0
    )
    lr_model = lr.fit(train_df)

    pred_lr = lr_model.transform(test_df)
    rmse_lr = evaluator_rmse.evaluate(pred_lr)
    r2_lr = evaluator_r2.evaluate(pred_lr)
    mae_real_lr, rmse_real_lr = evaluate_real_scale(pred_lr)

    print(f"[LR] RMSE(log): {rmse_lr:.4f} | R2(log): {r2_lr:.4f}")
    print(f"[LR] MAE(real): {mae_real_lr:,.2f} | RMSE(real): {rmse_real_lr:,.2f}")

    lr_out = os.path.join(MODEL_DIR, "lr_model")
    lr_model.write().overwrite().save(lr_out)
    print("✅ Saved LR model:", lr_out)

    # =========================
    # MODEL 2: GBT Regressor (mạnh vừa)
    # =========================
    print("\n🚀 Training GBTRegressor...")
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="label",
        maxIter=60,
        maxDepth=6,
        stepSize=0.1,
        subsamplingRate=0.8
    )
    gbt_model = gbt.fit(train_df)

    pred_gbt = gbt_model.transform(test_df)
    rmse_gbt = evaluator_rmse.evaluate(pred_gbt)
    r2_gbt = evaluator_r2.evaluate(pred_gbt)
    mae_real_gbt, rmse_real_gbt = evaluate_real_scale(pred_gbt)

    print(f"[GBT] RMSE(log): {rmse_gbt:.4f} | R2(log): {r2_gbt:.4f}")
    print(f"[GBT] MAE(real): {mae_real_gbt:,.2f} | RMSE(real): {rmse_real_gbt:,.2f}")

    gbt_out = os.path.join(MODEL_DIR, "gbt_model")
    gbt_model.write().overwrite().save(gbt_out)
    print("✅ Saved GBT model:", gbt_out)

    # 4) Chọn best theo RMSE(log)
    best_name = "LR" if rmse_lr <= rmse_gbt else "GBT"
    print(f"\n🏆 Best model by RMSE(log): {best_name}")

    spark.stop()


if __name__ == "__main__":
    main()
