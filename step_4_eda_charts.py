# step_4_eda_charts.py
"""
Bước 4 - EDA & Visualization.

MỤC TIÊU:
- Vẽ biểu đồ phân tích dữ liệu (Missing, Distribution, Correlation...).
- Đánh giá hiệu suất mô hình (Actual vs Predicted, Residuals).
- Xuất báo cáo ra file ảnh và text.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.regression import GBTRegressionModel, LinearRegressionModel


# ===== CẤU HÌNH ĐƯỜNG DẪN =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data2")

SILVER_PATH = os.path.join(DATA_DIR, "silver", "stackoverflow_silver_dev_clean.parquet")
GOLD_PATH = os.path.join(DATA_DIR, "salary_features_spark.parquet")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
REPORT_TXT = os.path.join(BASE_DIR, "reports", "eda_summary.txt")

# Đường dẫn model đã train
MODEL_GBT = os.path.join(BASE_DIR, "models", "gbt_model")
MODEL_LR = os.path.join(BASE_DIR, "models", "lr_model")

# Giới hạn số mẫu khi vẽ biểu đồ scatter để tránh lag máy
MAX_SAMPLE = 50000


def create_spark():
    spark = (
        SparkSession.builder
        .appName("EDA_Charts_Salary_BigData")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def ensure_dirs():
    """Tạo thư mục chứa ảnh báo cáo nếu chưa có"""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_TXT), exist_ok=True)


def save_fig(name: str):
    """Lưu biểu đồ hiện tại ra file ảnh"""
    path = os.path.join(FIG_DIR, name)
    try:
        plt.tight_layout() # Tự động căn chỉnh lề
    except Exception:
        pass
    plt.savefig(path, dpi=200) # dpi=200 cho ảnh nét
    plt.close()
    return path


def write_report(lines):
    """Ghi nội dung báo cáo ra file text"""
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def spark_quantiles(df, col, qs=(0.01, 0.5, 0.99), rel_err=0.001):
    """Tính phân vị (Quantile) bằng Spark (nhanh hơn Pandas với dữ liệu lớn)"""
    return df.approxQuantile(col, list(qs), rel_err)


def collect_sample(df, col, max_n=MAX_SAMPLE, seed=42):
    """Lấy mẫu ngẫu nhiên một cột về máy local để vẽ biểu đồ"""
    sample_df = df.select(col).where(F.col(col).isNotNull()).sample(False, 1.0, seed=seed).limit(max_n)
    return [r[col] for r in sample_df.collect()]


def bar_top_missing(df, total_rows, top_n=20):
    """Vẽ biểu đồ cột: Top các cột thiếu dữ liệu nhiều nhất"""
    exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    miss = df.select(exprs).collect()[0].asDict()

    miss_df = (
        pd.DataFrame({"col": list(miss.keys()), "missing": list(miss.values())})
        .assign(missing_pct=lambda x: x["missing"] / max(total_rows, 1) * 100)
        .sort_values("missing_pct", ascending=False)
        .head(top_n)
    )

    plt.figure()
    plt.barh(miss_df["col"][::-1], miss_df["missing_pct"][::-1])
    plt.xlabel("Missing (%)")
    plt.title(f"Top {top_n} cột thiếu dữ liệu (Missing Rate)")
    return save_fig("01_missing_top20.png")


def histogram_salary(df):
    """Vẽ biểu đồ phân phối lương (Histogram)"""
    # Lấy mẫu 99% (bỏ 1% lương cao nhất để biểu đồ không bị dẹt)
    q01, q50, q99 = spark_quantiles(df, "ConvertedCompYearly", (0.01, 0.5, 0.99))
    sample = collect_sample(df.where(F.col("ConvertedCompYearly") <= F.lit(q99)), "ConvertedCompYearly")

    plt.figure()
    plt.hist(sample, bins=60)
    plt.xlabel("ConvertedCompYearly")
    plt.ylabel("Count")
    plt.title("Phân phối lương (cắt ở 99% để dễ quan sát)")
    path1 = save_fig("02_salary_hist_raw.png")

    # Vẽ phân phối log(lương) -> Thường sẽ có hình chuông (Normal Distribution)
    sample_log = [math.log1p(x) for x in sample if x is not None and x > 0]
    plt.figure()
    plt.hist(sample_log, bins=60)
    plt.xlabel("log1p(ConvertedCompYearly)")
    plt.ylabel("Count")
    plt.title("Phân phối lương trên thang log (log1p)")
    path2 = save_fig("03_salary_hist_log.png")

    return (q01, q50, q99, path1, path2)


def plot_top_countries(df, top_n=15):
    """Vẽ biểu đồ Top quốc gia tham gia khảo sát"""
    if "CountryGrouped" not in df.columns:
        return None

    pdf = (
        df.groupBy("CountryGrouped")
        .agg(F.count("*").alias("n"))
        .orderBy(F.desc("n"))
        .limit(top_n)
        .toPandas()
    )

    plt.figure()
    plt.barh(pdf["CountryGrouped"][::-1], pdf["n"][::-1])
    plt.xlabel("Số lượng bản ghi")
    plt.title(f"Top {top_n} quốc gia/nhóm quốc gia theo số lượng mẫu")
    return save_fig("04_top_countries_count.png")


def plot_median_salary_by_group(df, group_col, top_n=15, min_count=300, fname=""):
    """Vẽ biểu đồ lương trung vị theo nhóm (VD: Theo bằng cấp, theo Remote...)"""
    if group_col not in df.columns:
        return None

    agg = (
        df.groupBy(group_col)
        .agg(
            F.count("*").alias("n"),
            F.expr("percentile_approx(ConvertedCompYearly, 0.5, 10000)").alias("median_salary"),
            F.expr("percentile_approx(ConvertedCompYearly, 0.25, 10000)").alias("p25"),
            F.expr("percentile_approx(ConvertedCompYearly, 0.75, 10000)").alias("p75"),
        )
        .filter(F.col("n") >= F.lit(min_count))
        .orderBy(F.desc("n"))
        .limit(top_n)
        .toPandas()
    )

    # Vẽ Error Bar (Median ở giữa, râu trên dưới là khoảng tứ phân vị IQR)
    x = np.arange(len(agg))
    med = agg["median_salary"].to_numpy()
    yerr_low = med - agg["p25"].to_numpy()
    yerr_high = agg["p75"].to_numpy() - med

    plt.figure()
    plt.errorbar(x, med, yerr=[yerr_low, yerr_high], fmt="o")
    plt.xticks(x, agg[group_col].astype(str), rotation=45, ha="right")
    plt.ylabel("Median salary (ConvertedCompYearly)")
    plt.title(f"Median salary theo {group_col} (kèm IQR)")
    return save_fig(fname)


def correlation_numeric(df):
    """Vẽ Heatmap tương quan giữa các biến số"""
    candidates = ["AgeNum_imp", "YearsCodeNum_imp", "YearsCodeProNum_imp", "CompTotal", "ConvertedCompYearly"]
    cols = [c for c in candidates if c in df.columns]
    if len(cols) < 2:
        return None

    sample_df = df.select(cols).dropna().sample(False, 1.0, seed=42).limit(MAX_SAMPLE).toPandas()
    corr = sample_df.corr(numeric_only=True)

    plt.figure()
    plt.imshow(corr.values)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.colorbar()
    plt.title("Ma trận tương quan (sample)")
    return save_fig("08_corr_heatmap.png")


def scatter_years_pro_vs_salary(df):
    """Vẽ Scatter Plot: Kinh nghiệm vs Lương"""
    col_x = "YearsCodeProNum_imp" if "YearsCodeProNum_imp" in df.columns else None
    if col_x is None:
        return None

    sample_df = (
        df.select(col_x, "ConvertedCompYearly")
        .dropna()
        .sample(False, 1.0, seed=42)
        .limit(MAX_SAMPLE)
        .toPandas()
    )

    x = sample_df[col_x].to_numpy()
    y = np.log1p(sample_df["ConvertedCompYearly"].to_numpy())

    plt.figure()
    plt.scatter(x, y, s=8, alpha=0.3)
    plt.xlabel(col_x)
    plt.ylabel("log1p(ConvertedCompYearly)")
    plt.title("Kinh nghiệm chuyên nghiệp vs Lương (log)")
    return save_fig("09_scatter_yearspro_salarylog.png")


def year_trend(df):
    """Vẽ xu hướng lương theo năm khảo sát"""
    if "SurveyYear" not in df.columns:
        return None

    pdf = (
        df.groupBy("SurveyYear")
        .agg(
            F.count("*").alias("n"),
            F.expr("percentile_approx(ConvertedCompYearly, 0.5, 10000)").alias("median_salary")
        )
        .orderBy("SurveyYear")
        .toPandas()
    )

    plt.figure()
    plt.plot(pdf["SurveyYear"], pdf["median_salary"], marker="o")
    plt.xlabel("SurveyYear")
    plt.ylabel("Median salary")
    plt.title("Xu hướng median salary theo năm khảo sát")
    return save_fig("10_trend_median_salary_by_year.png")


def model_diagnostics(df, model_path, model_name="gbt"):
    """
    Đánh giá mô hình (Model Diagnostics):
    - Actual vs Predicted: Xem dự đoán có khớp thực tế không.
    - Residuals: Xem sai số phân phối thế nào.
    """
    if not os.path.exists(model_path):
        return None

    # Load model đã train
    if model_name == "gbt":
        model = GBTRegressionModel.load(model_path)
    else:
        model = LinearRegressionModel.load(model_path)

    # Chuẩn bị dữ liệu test (Label log)
    ds = (
        df.select("features", "ConvertedCompYearly")
        .filter(F.col("ConvertedCompYearly").isNotNull() & (F.col("ConvertedCompYearly") > 0))
        .withColumn("label", F.log1p(F.col("ConvertedCompYearly")))
        .select("features", "label")
        .cache()
    )

    train_df, test_df = ds.randomSplit([0.8, 0.2], seed=42)
    pred = model.transform(test_df).select("label", "prediction")

    # Clip prediction để tránh giá trị quá vô lý khi vẽ biểu đồ
    lo, hi = pred.approxQuantile("label", [0.01, 0.99], 0.001)
    pred = pred.withColumn("prediction_clip", F.greatest(F.lit(lo), F.least(F.col("prediction"), F.lit(hi))))

    pdf = pred.sample(False, 1.0, seed=42).limit(MAX_SAMPLE).toPandas()

    y_true_log = pdf["label"].to_numpy()
    y_pred_log = pdf["prediction"].to_numpy()

    # 1) Actual vs Pred (log scale)
    plt.figure()
    plt.scatter(y_true_log, y_pred_log, s=8, alpha=0.3)
    plt.xlabel("Actual (log1p)")
    plt.ylabel("Predicted (log1p)")
    plt.title(f"Actual vs Predicted (log) - {model_name.upper()}")
    p1 = save_fig(f"11_actual_vs_pred_log_{model_name}.png")

    # 2) Actual vs Pred (real scale)
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(pdf["prediction_clip"].to_numpy())
    plt.figure()
    plt.scatter(y_true, y_pred, s=8, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Actual salary (log axis)")
    plt.ylabel("Pred salary (log axis)")
    plt.title(f"Actual vs Predicted (real, clipped) - {model_name.upper()}")
    p2 = save_fig(f"12_actual_vs_pred_real_{model_name}.png")

    # 3) Residual histogram (Phân phối phần dư)
    resid = (y_true_log - y_pred_log)
    plt.figure()
    plt.hist(resid, bins=60)
    plt.xlabel("Residual (actual_log - pred_log)")
    plt.ylabel("Count")
    plt.title(f"Residual distribution (log) - {model_name.upper()}")
    p3 = save_fig(f"13_residual_hist_{model_name}.png")

    return (p1, p2, p3)


def main():
    ensure_dirs()
    spark = create_spark()

    if not os.path.exists(SILVER_PATH):
        raise FileNotFoundError(f"[ERROR] Khong thay SILVER dataset: {SILVER_PATH}")

    df = spark.read.parquet(SILVER_PATH)

    if "ConvertedCompYearly" not in df.columns:
        raise ValueError("[ERROR] Silver thieu cot ConvertedCompYearly.")

    # Lọc salary hợp lệ
    df = df.filter(F.col("ConvertedCompYearly").isNotNull() & (F.col("ConvertedCompYearly") > 0)).cache()
    total = df.count()

    report_lines = []
    report_lines.append("=== EDA SUMMARY (Salary Prediction Project) ===")
    report_lines.append(f"Total usable rows (salary > 0): {total}")
    report_lines.append(f"Silver path: {SILVER_PATH}")

    # 1) Missingness
    p = bar_top_missing(df, total, top_n=20)
    report_lines.append(f"Saved: {p}")

    # 2) Salary distributions
    q01, q50, q99, p_raw, p_log = histogram_salary(df)
    report_lines.append(f"Salary quantiles: q01={q01:.2f}, median={q50:.2f}, q99={q99:.2f}")
    report_lines.append(f"Saved: {p_raw}")
    report_lines.append(f"Saved: {p_log}")

    # 3) Top countries
    p = plot_top_countries(df, top_n=15)
    if p: report_lines.append(f"Saved: {p}")

    # 4) Salary by group
    p = plot_median_salary_by_group(df, "RemoteWork", top_n=10, min_count=300, fname="05_median_by_RemoteWork.png")
    if p: report_lines.append(f"Saved: {p}")

    p = plot_median_salary_by_group(df, "EdLevel", top_n=12, min_count=300, fname="06_median_by_EdLevel.png")
    if p: report_lines.append(f"Saved: {p}")

    p = plot_median_salary_by_group(df, "OrgSize", top_n=12, min_count=300, fname="07_median_by_OrgSize.png")
    if p: report_lines.append(f"Saved: {p}")

    # 5) Correlation
    p = correlation_numeric(df)
    if p: report_lines.append(f"Saved: {p}")

    # 6) Scatter
    p = scatter_years_pro_vs_salary(df)
    if p: report_lines.append(f"Saved: {p}")

    # 7) Trend
    p = year_trend(df)
    if p: report_lines.append(f"Saved: {p}")

    # 8) Model diagnostics (Dùng dữ liệu Gold)
    if os.path.exists(GOLD_PATH):
        print(f"[INFO] Dang doc du lieu Gold cho Model Diagnostics: {GOLD_PATH}")
        df_gold = spark.read.parquet(GOLD_PATH)
        
        gbt_diag = model_diagnostics(df_gold, MODEL_GBT, model_name="gbt")
        if gbt_diag:
            for x in gbt_diag:
                report_lines.append(f"Saved: {x}")

        lr_diag = model_diagnostics(df_gold, MODEL_LR, model_name="lr")
        if lr_diag:
            for x in lr_diag:
                report_lines.append(f"Saved: {x}")
    else:
        print(f"[WARN] Khong tim thay Gold dataset ({GOLD_PATH}), bo qua Model Diagnostics.")

    write_report(report_lines)
    print("\n".join(report_lines))
    print(f"\n[DONE] Tat ca bieu do da xuat ra: {FIG_DIR}")
    print(f"[DONE] Summary: {REPORT_TXT}")

    spark.stop()


if __name__ == "__main__":
    main()
