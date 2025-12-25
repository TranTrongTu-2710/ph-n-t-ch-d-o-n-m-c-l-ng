"""
Ứng dụng Web dự đoán lương (Streamlit + PySpark).

MỤC TIÊU:
- Tạo giao diện người dùng thân thiện.
- Load Pipeline và Model đã huấn luyện từ Spark.
- Nhận input từ người dùng -> Biến đổi qua Pipeline -> Dự đoán bằng Model -> Hiển thị kết quả.
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.regression import GBTRegressionModel

# ===== CẤU HÌNH =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PIPELINE_PATH = os.path.join(MODEL_DIR, "feature_pipeline_model")
GBT_MODEL_PATH = os.path.join(MODEL_DIR, "gbt_model")

# ===== KHỞI TẠO SPARK =====
@st.cache_resource # Cache để không phải khởi động lại Spark mỗi lần reload trang
def get_spark():
    """Khởi tạo SparkSession chạy ngầm bên dưới Web App"""
    spark = (
        SparkSession.builder
        .appName("Salary_Prediction_WebApp")
        .master("local[*]") # Chạy local
        .config("spark.driver.memory", "2g") # Cấp 2GB RAM cho driver
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

# ===== LOAD MODELS =====
@st.cache_resource
def load_models(_spark):
    """Load Pipeline xử lý dữ liệu và Model dự đoán đã lưu trước đó"""
    if not os.path.exists(PIPELINE_PATH) or not os.path.exists(GBT_MODEL_PATH):
        return None, None
    
    # PipelineModel: Chứa logic biến đổi text -> vector
    pipeline_model = PipelineModel.load(PIPELINE_PATH)
    # GBTRegressionModel: Chứa logic dự đoán lương từ vector
    gbt_model = GBTRegressionModel.load(GBT_MODEL_PATH)
    return pipeline_model, gbt_model

# ===== GIAO DIỆN (UI) =====
st.set_page_config(page_title="Salary Predictor", page_icon="💰")

st.title("💰 IT Salary Prediction")
st.markdown("Dự đoán mức lương lập trình viên dựa trên dữ liệu StackOverflow Survey.")

spark = get_spark()
pipeline_model, gbt_model = load_models(spark)

if not pipeline_model or not gbt_model:
    st.error("❌ Không tìm thấy Model! Hãy chạy step_3 và train_regression_v2 trước.")
    st.stop()

# --- FORM NHẬP LIỆU ---
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        years_pro = st.number_input("Years of Professional Experience", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
        country = st.selectbox("Country", ["United States of America", "Germany", "United Kingdom", "India", "Canada", "France", "Brazil", "Poland", "Netherlands", "Australia", "Other"])
        ed_level = st.selectbox("Education Level", [
            "Bachelor’s degree (B.A., B.S., B.Eng., etc.)",
            "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",
            "Some college/university study without earning a degree",
            "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
            "Other doctoral degree (Ph.D., Ed.D., etc.)",
            "Professional degree (JD, MD, etc.)"
        ])

    with col2:
        remote = st.selectbox("Remote Work", ["Remote", "Hybrid (some remote, some in-person)", "In-person"])
        org_size = st.selectbox("Organization Size", [
            "20 to 99 employees", "100 to 499 employees", "10,000 or more employees",
            "1,000 to 4,999 employees", "2 to 9 employees", "10 to 19 employees",
            "500 to 999 employees", "5,000 to 9,999 employees", "Just me - I am a freelancer, sole proprietor, etc."
        ])
        
        # Chọn kỹ năng (Demo một số phổ biến)
        languages = st.multiselect("Languages Worked With", 
            ["Python", "JavaScript", "Java", "C#", "C++", "TypeScript", "SQL", "HTML/CSS", "Bash/Shell", "Go", "Rust", "PHP"])
        databases = st.multiselect("Databases Worked With", 
            ["PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Microsoft SQL Server", "MariaDB", "Elasticsearch", "Oracle"])

    submitted = st.form_submit_button("🚀 Predict Salary")

# --- LOGIC DỰ ĐOÁN ---
if submitted:
    # 1. Tạo DataFrame từ input người dùng
    input_data = [{
        "YearsCodeProNum_imp": float(years_pro), # Dùng tên cột khớp với lúc train
        "CountryGrouped": country,
        "EdLevel": ed_level,
        "RemoteWork": remote,
        "OrgSize": org_size,
        "LanguageHaveWorkedWith": ";".join(languages), # Nối lại thành chuỗi ngăn cách bởi ;
        "DatabaseHaveWorkedWith": ";".join(databases)
    }]
    
    input_df = spark.createDataFrame(input_data)
    
    # 2. Transform qua pipeline (Biến đổi text -> vector)
    try:
        features_df = pipeline_model.transform(input_df)
        
        # 3. Predict (Dự đoán log lương)
        prediction = gbt_model.transform(features_df)
        pred_log = prediction.select("prediction").collect()[0][0]
        
        # 4. Convert log -> real (Đổi về lương thật)
        salary_pred = np.expm1(pred_log)
        
        st.success(f"### 💵 Predicted Annual Salary: ${salary_pred:,.2f}")
        st.info(f"Log Value: {pred_log:.4f}")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
