"""
Bước 1 - Bronze + khám phá dữ liệu StackOverflow Survey 2021-2024.

MỤC TIÊU:
- Đọc dữ liệu thô (CSV) nặng nề.
- Chuẩn hóa sơ bộ (thêm năm, ép kiểu số).
- Lưu sang định dạng Parquet (Bronze Layer) để Spark đọc nhanh hơn ở các bước sau.
"""

import os
from pyspark.sql import SparkSession, functions as F, types as T


# ================== CẤU HÌNH ĐƯỜNG DẪN ==================
# Sử dụng os.path.join để đường dẫn hoạt động đúng trên cả Windows/Linux/Mac
RAW_DATA_DIR = os.path.join("data2", "raw")
BRONZE_DIR = os.path.join("data2", "bronze")

# Map năm -> tên file CSV tương ứng
FILE_MAP = {
    2021: "survey_results_public_2021.csv",
    2022: "survey_results_public_2022.csv",
    2023: "survey_results_public_2023.csv",
    2024: "survey_results_public_2024.csv",
}


# ================== HÀM HỖ TRỢ ==================

def create_spark_session() -> SparkSession:
    """
    Khởi tạo SparkSession - điểm bắt đầu của mọi ứng dụng Spark.
    """
    spark = (
        SparkSession.builder
        .appName("so_survey_step_1_bronze_explore")
        # local[*]: Chạy trên máy cá nhân, dùng tất cả các nhân CPU đang có
        .master("local[*]")
        # LEGACY: Để Spark xử lý định dạng ngày tháng kiểu cũ nếu có lỗi parse
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    # Chỉ hiện cảnh báo (WARN) hoặc lỗi (ERROR), ẩn các log thông tin (INFO) rác
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_one_year(spark: SparkSession, year: int, filename: str):
    """
    Đọc 1 file CSV, xử lý sơ bộ và trả về DataFrame.
    """
    path = os.path.join(RAW_DATA_DIR, filename)
    print(f"[INFO] Dang doc file nam {year}: {path}")

    # Đọc CSV với các tùy chọn quan trọng
    df = (
        spark.read
        .option("header", True)       # Dòng đầu tiên là tên cột
        .option("inferSchema", True)  # Spark tự đoán kiểu dữ liệu (int, string...)
        .option("multiLine", True)    # Cho phép 1 dòng dữ liệu xuống dòng (quan trọng với cột text dài)
        .option("escape", "\"")       # Xử lý ký tự đặc biệt trong chuỗi
        .csv(path)
    )

    # Thêm cột SurveyYear để sau này gộp nhiều năm lại vẫn biết dòng nào của năm nào
    df = df.withColumn("SurveyYear", F.lit(year))

    # Ép kiểu cột lương về số thực (Double) để tránh lỗi tính toán sau này
    for col_name in ["ConvertedCompYearly", "CompTotal"]:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(T.DoubleType()))

    return df


def show_basic_stats(df, year: int):
    """
    In ra thống kê nhanh để kiểm tra dữ liệu có ổn không.
    """
    print(f"\n===== Thong ke nhanh cho nam {year} =====")
    total_rows = df.count()
    print(f"Tong so dong: {total_rows}")

    # Kiểm tra cột lương (Target Variable)
    if "ConvertedCompYearly" in df.columns:
        salary_not_null = df.filter(F.col("ConvertedCompYearly").isNotNull())
        salary_null = total_rows - salary_not_null.count()

        print(f"So dong co luong: {salary_not_null.count()}")
        print(f"So dong thieu luong: {salary_null}")

        # Tính trung bình và trung vị (median) lương
        if salary_not_null.count() > 0:
            agg = (
                salary_not_null
                    .agg(
                        F.mean("ConvertedCompYearly").alias("mean_salary"),
                        # percentile_approx(0.5) chính là Median
                        F.expr("percentile_approx(ConvertedCompYearly, 0.5)").alias("median_salary")
                    )
                    .collect()[0]
            )
            print(f"Luong trung binh: {agg['mean_salary']:.2f}")
            print(f"Luong median:    {agg['median_salary']:.2f}")
    else:
        print("[WARN] Khong tim thay cot ConvertedCompYearly.")


def save_bronze(df, year: int):
    """
    Lưu DataFrame ra file Parquet.
    Parquet nén tốt hơn CSV và giữ được kiểu dữ liệu (schema).
    """
    out_path = os.path.join(BRONZE_DIR, f"stackoverflow_{year}")
    print(f"[INFO] Ghi du lieu nam {year} vao: {out_path}")

    (
        df.write
        .mode("overwrite")  # Ghi đè nếu file đã tồn tại
        .parquet(out_path)
    )


def union_all_years(dfs):
    """
    Gộp dữ liệu các năm lại thành 1 DataFrame lớn.
    """
    it = iter(dfs.values())
    df_all = next(it)
    for df in it:
        # allowMissingColumns=True: Nếu năm 2021 thiếu cột mà 2022 có, Spark vẫn cho gộp (điền null)
        df_all = df_all.unionByName(df, allowMissingColumns=True)
    return df_all


# ================== MAIN ==================

def main():
    # Tạo thư mục chứa dữ liệu nếu chưa có
    os.makedirs(BRONZE_DIR, exist_ok=True)

    spark = create_spark_session()
    dfs_by_year = {}

    # 1. Vòng lặp xử lý từng năm
    for year, filename in FILE_MAP.items():
        df_year = load_one_year(spark, year, filename)
        dfs_by_year[year] = df_year

        show_basic_stats(df_year, year)
        save_bronze(df_year, year) # Lưu ngay sau khi đọc

    # 2. Thử gộp lại để xem thống kê tổng quan
    print("\n===== Thong ke tong hop luong theo nam (tu Bronze) =====")
    df_all = union_all_years(dfs_by_year)

    if "ConvertedCompYearly" in df_all.columns:
        stats_by_year = (
            df_all
            .groupBy("SurveyYear")
            .agg(
                F.count("*").alias("n_rows"),
                F.count(F.col("ConvertedCompYearly")).alias("n_salary_not_null"),
                F.mean("ConvertedCompYearly").alias("mean_salary"),
                F.expr("percentile_approx(ConvertedCompYearly, 0.5)").alias("median_salary"),
            )
            .orderBy("SurveyYear")
        )
        stats_by_year.show(truncate=False)

    spark.stop()
    print("\n[DONE] Hoan thanh Buoc 1 (Bronze).")


if __name__ == "__main__":
    main()
