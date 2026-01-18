# Sales Prediction Using Trend Analysis

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen)](https://sales-prediction-trend-analysis-ttea4fyaqz4a8g77yhtxzr.streamlit.app)

---

## 🔗 Live Deployment
https://sales-prediction-trend-analysis-ttea4fyaqz4a8g77yhtxzr.streamlit.app

---

## 📌 Project Overview
This project focuses on analyzing historical sales data to identify trends and forecast future sales using time-series analysis techniques. The application is built using Python and Streamlit, allowing users to upload datasets, visualize trends, and generate future sales predictions interactively.

The solution is designed to support business decision-making such as inventory planning, demand estimation, and revenue forecasting.

---

## 🎯 Objectives
- Analyze historical sales patterns
- Identify long-term sales trends
- Forecast future sales using statistical models
- Provide actionable business insights

---

## 📊 Dataset
**Source:** Kaggle (Retail Sales Datasets)

Supported columns:
- Date / Order Date
- Amount / Sales / Sales_Amount / Weekly_Sales

The data is aggregated at a monthly level to ensure stable and reliable forecasting.

---

## 🧠 Methodology

### 1. Data Preprocessing
- Automatic column detection
- Date conversion and validation
- Numerical data cleaning
- Monthly sales aggregation

### 2. Trend Analysis
- Visualization of historical sales trends
- Identification of growth or decline patterns

### 3. Forecasting Models
- **Linear Regression** – baseline trend model
- **Holt–Winters Exponential Smoothing** – primary forecasting model

### 4. Model Evaluation
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)

---

## 📈 Results
- Holt–Winters model produced smoother and more reliable forecasts
- Forecasting supports proactive business planning
- Interactive dashboard improves interpretability

---

## 💼 Business Insights
- Increasing trend suggests inventory expansion opportunities
- Declining trend indicates need for promotional strategies
- Forecasting aids data-driven decision-making

---

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- Streamlit

---

## ▶️ How to Run Locally

1. Install dependencies:

2. Run the application:

3. Upload a sales dataset (CSV or Excel)

---

## 👨‍💻 Author
**Mukeshram S**  
B.Tech – Computer Science Engineering (AI & Data Science)

---

## 📎 Notes
- This project is developed for academic and internship learning purposes.
- Dataset used is publicly available from Kaggle.
