"""
Bước 2 - Silver: Làm sạch và Chuẩn hóa dữ liệu.

MỤC TIÊU:
- Lọc dữ liệu: Chỉ lấy Developer chuyên nghiệp (bỏ sinh viên, người làm hobby).
- Xử lý Outlier: Bỏ những mức lương quá ảo (quá thấp hoặc quá cao).
- Feature Engineering:
    + Chuyển đổi cột chuỗi "10-20 years" thành số.
    + Gom nhóm các quốc gia ít mẫu thành "Other".
    + Điền dữ liệu thiếu (Imputation).
"""

import os
import re
from typing import Dict

from pyspark.sql import SparkSession, DataFrame, functions as F, types as T
from pyspark.ml.feature import Imputer


# ================== CẤU HÌNH ==================
BRONZE_DIR = os.path.join("data2", "bronze")
SILVER_DIR = os.path.join("data2", "silver")
YEARS = [2021, 2022, 2023, 2024]


def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("so_survey_step_2_silver_prepare")
        .master("local[*]")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_bronze_year(spark: SparkSession, year: int) -> DataFrame:
    """Đọc dữ liệu Bronze đã lưu ở bước 1"""
    path = os.path.join(BRONZE_DIR, f"stackoverflow_{year}")
    print(f"📥 Đang đọc bronze năm {year} từ: {path}")
    df = spark.read.parquet(path)
    # Phòng hờ nếu bước 1 quên thêm SurveyYear
    if "SurveyYear" not in df.columns:
        df = df.withColumn("SurveyYear", F.lit(year))
    return df


def union_all_years(dfs_by_year: Dict[int, DataFrame]) -> DataFrame:
    """Gộp 4 năm lại thành 1 DataFrame duy nhất"""
    it = iter(dfs_by_year.values())
    df_all = next(it)
    for df in it:
        df_all = df_all.unionByName(df, allowMissingColumns=True)
    return df_all


def build_dev_population(df: DataFrame) -> DataFrame:
    """
    LỌC DỮ LIỆU (FILTERING):
    Chỉ giữ lại những dòng thỏa mãn tiêu chí là "Developer chuyên nghiệp".
    """
    cond = F.lit(True)

    # 1. MainBranch phải chứa chữ "developer"
    if "MainBranch" in df.columns:
        cond = cond & F.lower(F.col("MainBranch")).contains("developer")

    # 2. Employment phải là "employed" (đang đi làm)
    if "Employment" in df.columns:
        cond = cond & F.lower(F.col("Employment")).contains("employed")

    # 3. DevType phải chứa các từ khóa liên quan đến lập trình
    if "DevType" in df.columns:
        keywords = ["developer", "engineer", "devops", "sre", "data scientist", "machine learning"]
        pattern = "(?i)(" + "|".join(keywords) + ")" # Regex: (?i) nghĩa là không phân biệt hoa thường
        cond = cond & F.col("DevType").rlike(pattern)

    # 4. Phải có thông tin về lương (Target không được null)
    if "ConvertedCompYearly" in df.columns:
        cond = cond & F.col("ConvertedCompYearly").isNotNull()

    df_filtered = df.filter(cond)
    print("👨‍💻 Số dòng sau khi lọc dev ngành phần mềm:", df_filtered.count())
    return df_filtered


def clean_salary_outliers(df: DataFrame) -> DataFrame:
    """
    LOẠI BỎ OUTLIER (Nhiễu):
    Lương quá thấp (ví dụ $1/năm) hoặc quá cao (ví dụ $100M/năm) sẽ làm hỏng mô hình.
    Dùng phương pháp Quantile để cắt bỏ 1% đầu và 1% đuôi.
    """
    if "ConvertedCompYearly" not in df.columns:
        return df

    # Tính ngưỡng 1% và 99%
    low, high = df.approxQuantile("ConvertedCompYearly", [0.01, 0.99], 0.01)
    print(f"➡️ Giữ lại lương trong khoảng [{low:.2f}, {high:.2f}]")
    
    df_filtered = df.filter(
        (F.col("ConvertedCompYearly") >= F.lit(low)) &
        (F.col("ConvertedCompYearly") <= F.lit(high))
    )
    print("💾 Số dòng sau khi loại outlier lương:", df_filtered.count())
    return df_filtered


# ---------- UDF (User Defined Function) để xử lý chuỗi ----------

def _parse_years_code(value: str):
    """
    Chuyển đổi chuỗi kinh nghiệm thành số.
    VD: 'Less than 1 year' -> 0.5
        'More than 50 years' -> 50.0
        '10' -> 10.0
    """
    if value is None: return None
    v = str(value).strip()
    if v == "" or v.lower() == "na": return None
    lower = v.lower()
    
    if "less than 1 year" in lower: return 0.5
    if "more than" in lower:
        nums = re.findall(r"\d+", v) # Tìm số trong chuỗi
        return float(nums[0]) if nums else None
    
    # Trường hợp bình thường (là số)
    try:
        return float(v)
    except:
        # Trường hợp lỗi khác, cố gắng tìm số đầu tiên
        nums = re.findall(r"\d+", v)
        return float(nums[0]) if nums else None

# Đăng ký hàm Python thành hàm Spark UDF
parse_years_code_udf = F.udf(_parse_years_code, T.DoubleType())


def _parse_age_to_numeric(age_str: str):
    """
    Chuyển đổi khoảng tuổi thành số trung bình.
    VD: '25-34 years old' -> 29.5
    """
    if age_str is None: return None
    s = str(age_str)
    if "Under 18" in s: return 17.0
    if "65 years or older" in s: return 65.0
    
    nums = re.findall(r"\d+", s)
    if not nums: return None
    
    if len(nums) == 1: return float(nums[0])
    
    # Lấy trung bình cộng của khoảng (VD: 25 và 34)
    a, b = float(nums[0]), float(nums[1])
    return (a + b) / 2.0

parse_age_udf = F.udf(_parse_age_to_numeric, T.DoubleType())


def group_small_countries(df: DataFrame, min_count: int = 100) -> DataFrame:
    """
    Gom các quốc gia có ít hơn 100 mẫu thành nhóm 'Other'.
    Giúp giảm số lượng biến (dimension reduction) khi One-Hot Encoding sau này.
    """
    if "Country" not in df.columns: return df
    
    country_counts = df.groupBy("Country").count()
    
    # Tìm danh sách các nước nhỏ
    small_countries = [
        row["Country"] for row in country_counts
        .filter(F.col("count") < F.lit(min_count))
        .collect()
    ]
    
    if not small_countries:
        return df.withColumn("CountryGrouped", F.col("Country"))

    # Thay thế tên nước bằng 'Other' nếu nằm trong danh sách nhỏ
    df_grouped = df.withColumn(
        "CountryGrouped",
        F.when(F.col("Country").isin(small_countries), F.lit("Other"))
         .otherwise(F.col("Country"))
    )
    return df_grouped


def impute_numeric_features(df: DataFrame) -> DataFrame:
    """
    Điền dữ liệu thiếu (Missing Value Imputation) cho các cột số.
    Chiến lược: Dùng giá trị trung vị (median) của cột đó.
    """
    numeric_cols = []
    for c in ["YearsCodeProNum", "YearsCodeNum", "AgeNum"]:
        if c in df.columns: numeric_cols.append(c)

    if not numeric_cols: return df

    # Tạo tên cột mới có hậu tố _imp
    output_cols = [c + "_imp" for c in numeric_cols]
    print("🛠 Đang impute median cho:", numeric_cols)
    
    imputer = Imputer(inputCols=numeric_cols, outputCols=output_cols).setStrategy("median")
    model = imputer.fit(df)
    return model.transform(df)


def clean_text_columns(df: DataFrame) -> DataFrame:
    """
    Điền giá trị rỗng cho các cột kỹ năng (Language, Database) để tránh lỗi NullPointerException.
    """
    text_cols = ["LanguageHaveWorkedWith", "DatabaseHaveWorkedWith"]
    fill_dict = {}
    for c in text_cols:
        if c in df.columns:
            fill_dict[c] = "None" # Nếu null thì điền chuỗi "None"
    
    if fill_dict:
        print(f"🧹 FillNA cho các cột text: {list(fill_dict.keys())}")
        df = df.fillna(fill_dict)
    
    return df


# ================== MAIN ==================

def main():
    os.makedirs(SILVER_DIR, exist_ok=True)
    spark = create_spark_session()

    # 1. Đọc và gộp dữ liệu 4 năm
    dfs_by_year = {year: load_bronze_year(spark, year) for year in YEARS}
    df_all_raw = union_all_years(dfs_by_year)

    # 2. Chọn các cột cần thiết (Feature Selection)
    core_cols = [
        "SurveyYear", "ConvertedCompYearly", "CompTotal", "Country",
        "MainBranch", "DevType", "Employment",
        "YearsCodePro", "YearsCode", "Age", "EdLevel", "OrgSize", "RemoteWork",
        "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith" # Cột kỹ năng mới thêm
    ]
    # Chỉ lấy những cột thực sự tồn tại trong dữ liệu
    available_core_cols = [c for c in core_cols if c in df_all_raw.columns]
    df_core = df_all_raw.select(*available_core_cols)

    # 3. Lọc Dev và loại bỏ Outlier lương
    df_dev = build_dev_population(df_core)
    df_dev = clean_salary_outliers(df_dev)

    # 4. Áp dụng UDF để chuyển đổi cột số
    if "YearsCodePro" in df_dev.columns:
        df_dev = df_dev.withColumn("YearsCodeProNum", parse_years_code_udf(F.col("YearsCodePro")))
    if "YearsCode" in df_dev.columns:
        df_dev = df_dev.withColumn("YearsCodeNum", parse_years_code_udf(F.col("YearsCode")))
    if "Age" in df_dev.columns:
        df_dev = df_dev.withColumn("AgeNum", parse_age_udf(F.col("Age")))

    # 5. Gom nhóm Country và Điền dữ liệu thiếu
    df_dev = group_small_countries(df_dev, min_count=100)
    df_dev = impute_numeric_features(df_dev)

    # 6. Xử lý cột Text (Kỹ năng)
    df_dev = clean_text_columns(df_dev)

    # 7. Lưu kết quả ra Silver
    out_path = os.path.join(SILVER_DIR, "stackoverflow_silver_dev_clean.parquet")
    print(f"💾 Ghi dataset Silver vào: {out_path}")
    
    df_dev.write.mode("overwrite").parquet(out_path)

    spark.stop()
    print("\n✅ Hoàn thành Bước 2 - Silver.")


if __name__ == "__main__":
    main()
