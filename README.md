# Sales Prediction Using Trend Analysis

## Project Overview
This project focuses on analyzing historical sales data to identify trends and forecast future sales using time-series techniques. The application is developed using Python and Streamlit, enabling interactive analysis and visualization.

The goal is to assist businesses in understanding sales behavior and supporting data-driven decision-making for inventory planning and revenue forecasting.

---

## Objectives
- Analyze historical sales patterns
- Identify overall sales trend
- Forecast future sales using statistical models
- Provide business insights based on predictions

---

## Dataset
Source: Kaggle (Retail Sales Dataset)

Key Columns Used:
- Date: Transaction or sales date
- Amount / Sales_Amount / Weekly_Sales: Revenue values

The dataset is aggregated at a monthly level to ensure accurate time-series forecasting.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- Streamlit

---

## Methodology

### 1. Data Preprocessing
- Converted date column to datetime format
- Removed missing or invalid records
- Cleaned numerical sales values
- Aggregated daily/weekly sales into monthly totals

### 2. Exploratory Analysis
- Visualized monthly sales trends
- Observed growth and fluctuation patterns

### 3. Forecasting Models
Two models were implemented:

**Linear Regression**
- Used as a baseline trend model
- Captures long-term sales direction

**Holt–Winters Exponential Smoothing**
- Captures trend components
- Suitable for retail sales forecasting

### 4. Forecasting
- Predicted sales for the next 3–12 months
- Compared forecasts from both models

---

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)

These metrics help measure prediction accuracy.

---

## Results
- The model identifies overall sales direction
- Holt–Winters provides smoother and more reliable forecasts
- Forecast results assist in inventory and revenue planning

---

## Business Insights
- Upward trend indicates scope for inventory expansion
- Downward or stable trend suggests promotional strategies
- Forecasting supports proactive decision-making

---

## Conclusion
This project demonstrates the application of time-series analysis in real-world business scenarios. It highlights how historical sales data can be transformed into actionable insights using statistical forecasting techniques.

---

## How to Run
1. Install dependencies:

2. Run the application:

3. Upload a sales dataset (CSV or Excel)

---

## Author
Mukeshram S  
B.Tech CSE (AI & DS)
