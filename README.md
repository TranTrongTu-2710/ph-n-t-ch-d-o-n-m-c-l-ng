# 📊 Phân tích & Dự đoán Mức Lương Ngành CNTT (Big Data + Spark)

## 🔍 Giới thiệu
Dự án tập trung vào phân tích và dự đoán mức lương ngành Công nghệ Thông tin
dựa trên dữ liệu khảo sát Stack Overflow (2021–2024), sử dụng Apache Spark
để xử lý dữ liệu lớn và Machine Learning để xây dựng mô hình dự báo.

## 🧱 Kiến trúc dự án
- Apache Spark (PySpark) – xử lý dữ liệu lớn
- Spark MLlib – huấn luyện mô hình (GBTRegressor)
- FastAPI – xây dựng web demo dự đoán
- Matplotlib – trực quan hóa dữ liệu

## 📁 Cấu trúc thư mục
dll/
├── app/                         # Web FastAPI
│   ├── main.py
│   └── templates/
│       └── index.html
│
├── data2/                       # Dữ liệu (KHÔNG push raw lớn)
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── models/                      # Model Spark (KHÔNG push)
│   └── salary_pipeline_model/
│
├── reports/
│   └── figures/                # Các biểu đồ PNG (NÊN push)
│
├── step_1_bronze_explore.py
├── step_2_silver_prepare.py
├── step_3_gold_features.py
├── train_pipeline_spark.py
├── train_regression_spark.py
│
├── requirements.txt
├── README.md
└── .gitignore

## 🔄 Quy trình thực hiện
1. Bronze: Khám phá dữ liệu gốc
2. Silver: Làm sạch, chuẩn hóa dữ liệu
3. Gold: Feature engineering
4. Train: Huấn luyện & đánh giá mô hình
5. Web: Demo dự đoán lương qua giao diện web

## 📈 Một số kết quả
- RMSE (log): ~0.93
- R² (log): ~0.46
- GBT cho kết quả tốt hơn Linear Regression

## 🌐 Demo Web
Web cho phép người dùng nhập:
- Kinh nghiệm
- Hình thức làm việc (Remote/Hybrid/Onsite)
- Trình độ học vấn
- Quy mô công ty  
→ Dự đoán mức lương tương ứng.

## 👨‍💻 Tác giả
- **Trần Trọng Tú**
- GitHub: https://github.com/TranTrongTu-2710
