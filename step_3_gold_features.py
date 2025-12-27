# step_3_gold_features.py
"""
Bước 3 - GOLD: Feature Engineering & Pipeline Construction.

MỤC TIÊU:
- Biến đổi dữ liệu dạng chữ (Category, Text) thành dạng số (Vector).
- Xây dựng Pipeline xử lý tự động.
- Lưu Pipeline Model để tái sử dụng cho Web App.
"""

import os
from typing import List

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler,
    RegexTokenizer, CountVectorizer
)


# ====== CẤU HÌNH ĐƯỜNG DẪN ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data2")
MODEL_DIR = os.path.join(BASE_DIR, "models")

SILVER_PATH = os.path.join(DATA_DIR, "silver", "stackoverflow_silver_dev_clean.parquet")
FEATURES_PATH = os.path.join(DATA_DIR, "salary_features_spark.parquet")
PIPELINE_MODEL_PATH = os.path.join(MODEL_DIR, "feature_pipeline_model")


def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("so_survey_step_3_gold_features_advanced")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def choose_numeric_columns(df: DataFrame) -> List[str]:
    """Chọn cột số: Chỉ dùng YearsCodePro (Kinh nghiệm chuyên nghiệp)"""
    # Ưu tiên cột đã được impute (điền thiếu)
    candidates = ["YearsCodeProNum_imp", "YearsCodeProNum"]
    for c in candidates:
        if c in df.columns:
            print(f"[INFO] Numeric feature duoc chon: {c}")
            return [c]
    print("[WARN] Khong tim thay YearsCodeProNum!")
    return []


def choose_categorical_columns(df: DataFrame) -> List[str]:
    """Chọn cột phân loại (Categorical)"""
    candidate_cats = [
        "CountryGrouped", # Quốc gia (đã gom nhóm)
        "EdLevel",        # Trình độ học vấn
        "OrgSize",        # Quy mô công ty
        "RemoteWork",     # Làm từ xa hay tại văn phòng
    ]
    categorical_cols = [c for c in candidate_cats if c in df.columns]
    print("[INFO] Categorical features duoc chon:", categorical_cols)
    return categorical_cols


def choose_text_columns(df: DataFrame) -> List[str]:
    """Chọn cột văn bản (Text) chứa danh sách kỹ năng"""
    candidates = ["LanguageHaveWorkedWith", "DatabaseHaveWorkedWith"]
    text_cols = [c for c in candidates if c in df.columns]
    print("[INFO] Text features duoc chon:", text_cols)
    return text_cols


def build_pipeline(categorical_cols: List[str], numeric_cols: List[str], text_cols: List[str]) -> Pipeline:
    """
    Xây dựng chuỗi xử lý (Pipeline) gồm nhiều bước nối tiếp nhau.
    """
    stages = []
    assembler_inputs = []

    # 1. Xử lý Categorical: StringIndexer -> OneHotEncoder
    # StringIndexer: Biến đổi chuỗi thành chỉ số (VD: USA -> 0, India -> 1)
    # OneHotEncoder: Biến đổi chỉ số thành vector nhị phân (VD: 0 -> [1,0,0], 1 -> [0,1,0])
    for c in categorical_cols:
        idx_col = f"{c}_idx"
        vec_col = f"{c}_vec"
        
        # handleInvalid="keep": Nếu gặp giá trị lạ chưa từng thấy lúc train, vẫn giữ lại (thành vector 0 hết)
        stages.append(StringIndexer(inputCol=c, outputCol=idx_col, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[idx_col], outputCols=[vec_col], handleInvalid="keep"))
        assembler_inputs.append(vec_col)

    # 2. Xử lý Text: RegexTokenizer -> CountVectorizer
    # RegexTokenizer: Tách chuỗi "Python;Java" thành mảng ["Python", "Java"] dựa trên dấu chấm phẩy
    # CountVectorizer: Đếm tần suất xuất hiện của từ khóa (Bag of Words)
    for c in text_cols:
        tok_col = f"{c}_tokens"
        vec_col = f"{c}_counts"
        
        stages.append(RegexTokenizer(inputCol=c, outputCol=tok_col, pattern=r";"))
        # vocabSize=30: Chỉ lấy Top 30 từ phổ biến nhất để tránh vector quá lớn
        stages.append(CountVectorizer(inputCol=tok_col, outputCol=vec_col, vocabSize=30, minDF=0.01))
        assembler_inputs.append(vec_col)

    # 3. Thêm cột số vào danh sách đầu vào
    assembler_inputs.extend(numeric_cols)

    # 4. VectorAssembler: Gom tất cả các cột (số, one-hot, text vector) thành 1 vector duy nhất gọi là "features_raw"
    print("[INFO] Cac cot dau vao cho VectorAssembler:", assembler_inputs)
    stages.append(VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features_raw",
        handleInvalid="keep"
    ))

    # 5. StandardScaler: Chuẩn hóa dữ liệu về cùng một thang đo (Mean=0, Std=1)
    # Giúp thuật toán Linear Regression hội tụ nhanh hơn và chính xác hơn
    stages.append(StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=False, # Giữ nguyên tính thưa (sparsity) của vector để tiết kiệm RAM
        withStd=True
    ))

    return Pipeline(stages=stages)


def main():
    spark = create_spark_session()

    if not os.path.exists(SILVER_PATH):
        raise FileNotFoundError(f"[ERROR] Khong tim thay SILVER_PATH: {SILVER_PATH}")

    print(f"[INFO] Dang doc du lieu Silver tu: {SILVER_PATH}")
    df = spark.read.parquet(SILVER_PATH)

    # Lọc bỏ dòng thiếu lương hoặc lương <= 0
    if "ConvertedCompYearly" not in df.columns:
        raise ValueError("[ERROR] Thieu cot ConvertedCompYearly")
    
    df = df.filter(F.col("ConvertedCompYearly").isNotNull() & (F.col("ConvertedCompYearly") > 0))

    # Chọn features
    numeric_cols = choose_numeric_columns(df)
    categorical_cols = choose_categorical_columns(df)
    text_cols = choose_text_columns(df)

    # Chỉ giữ lại cột cần thiết
    keep_cols = ["ConvertedCompYearly"] + numeric_cols + categorical_cols + text_cols
    df = df.select(*keep_cols)
    df = df.cache() # Cache vào RAM để chạy nhanh hơn
    print(f"[INFO] So luong mau training: {df.count()}")

    # Xây dựng và Huấn luyện Pipeline (Fit)
    # Lúc này Spark sẽ học từ điển (Vocabulary) và các chỉ số (String Index) từ dữ liệu
    pipeline = build_pipeline(categorical_cols, numeric_cols, text_cols)
    model = pipeline.fit(df)

    # Biến đổi dữ liệu (Transform)
    out = model.transform(df).select("features", "ConvertedCompYearly")

    # Ghi dữ liệu đã biến đổi ra file (để train model ở bước sau)
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    out.repartition(32).write.mode("overwrite").parquet(FEATURES_PATH)
    print(f"[INFO] Da ghi GOLD features ra: {FEATURES_PATH}")

    # Ghi Pipeline Model ra file (để Web App dùng lại)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.write().overwrite().save(PIPELINE_MODEL_PATH)
    print(f"[INFO] Da luu Feature Pipeline Model ra: {PIPELINE_MODEL_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
