import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Sales Prediction Using Trend Analysis",
    layout="wide"
)

st.title("Sales Prediction Using Trend Analysis")
st.write("Time-series based sales forecasting and trend analysis")

# ======================================================
# FILE UPLOAD
# ======================================================

uploaded_file = st.file_uploader(
    "Upload Sales Dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # ======================================================
    # LOAD DATA
    # ======================================================

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ======================================================
    # COLUMN STANDARDIZATION
    # ======================================================

    df.columns = df.columns.str.strip()

    amount_aliases = [
        "Amount", "Sales", "Sales_Amount",
        "Revenue", "Total_Sales", "Weekly_Sales"
    ]

    date_aliases = [
        "Date", "date", "Order Date",
        "order_date", "Transaction Date"
    ]

    amount_col = None
    date_col = None

    for col in df.columns:
        if col in amount_aliases:
            amount_col = col
        if col in date_aliases:
            date_col = col

    if amount_col is None or date_col is None:
        st.error("Required columns not detected.")
        st.write("Detected columns:", list(df.columns))
        st.stop()

    df = df.rename(columns={
        amount_col: "Amount",
        date_col: "Date"
    })

    # ======================================================
    # DATA CLEANING
    # ======================================================

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df["Amount"] = (
        df["Amount"]
        .astype(str)
        .str.replace(",", "")
        .str.replace("₹", "")
        .astype(float)
    )

    # ======================================================
    # MONTHLY AGGREGATION
    # ======================================================

    monthly_sales = (
        df.set_index("Date")
        .resample("M")["Amount"]
        .sum()
        .reset_index()
    )

    st.subheader("Monthly Aggregated Sales")
    st.dataframe(monthly_sales)

    # ======================================================
    # TREND VISUALIZATION
    # ======================================================

    st.subheader("Monthly Sales Trend")

    fig, ax = plt.subplots()
    ax.plot(monthly_sales["Date"], monthly_sales["Amount"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales Amount")
    st.pyplot(fig)

    # ======================================================
    # FORECAST SETTINGS
    # ======================================================

    forecast_period = st.slider(
        "Select forecast period (months)",
        3, 12, 6
    )

    # ======================================================
    # LINEAR REGRESSION MODEL
    # ======================================================

    monthly_sales["TimeIndex"] = np.arange(len(monthly_sales))

    X = monthly_sales[["TimeIndex"]]
    y = monthly_sales["Amount"]

    lr_model = LinearRegression()
    lr_model.fit(X, y)

    future_index = np.arange(
        len(monthly_sales),
        len(monthly_sales) + forecast_period
    ).reshape(-1, 1)

    lr_forecast = lr_model.predict(future_index)

    # ======================================================
    # HOLT-WINTERS MODEL
    # ======================================================

    hw_model = ExponentialSmoothing(
        monthly_sales["Amount"],
        trend="add",
        seasonal=None
    ).fit()

    hw_forecast = hw_model.forecast(forecast_period)

    # ======================================================
    # FORECAST RESULTS
    # ======================================================

    forecast_dates = pd.date_range(
        start=monthly_sales["Date"].iloc[-1],
        periods=forecast_period + 1,
        freq="M"
    )[1:]

    forecast_df = pd.DataFrame({
        "Month": forecast_dates,
        "Linear Regression Forecast": lr_forecast,
        "Holt-Winters Forecast": hw_forecast.values
    })

    st.subheader("Forecast Results")
    st.dataframe(forecast_df)

    # ======================================================
    # FORECAST VISUALIZATION
    # ======================================================

    fig2, ax2 = plt.subplots()
    ax2.plot(
        monthly_sales["Date"],
        monthly_sales["Amount"],
        label="Historical"
    )
    ax2.plot(
        forecast_df["Month"],
        forecast_df["Linear Regression Forecast"],
        linestyle="--",
        label="Linear Regression"
    )
    ax2.plot(
        forecast_df["Month"],
        forecast_df["Holt-Winters Forecast"],
        linestyle="--",
        label="Holt-Winters"
    )
    ax2.legend()
    st.pyplot(fig2)

    # ======================================================
    # MODEL EVALUATION
    # ======================================================

    st.subheader("Model Evaluation")

    actual = monthly_sales["Amount"][-forecast_period:]

    lr_eval = lr_forecast[:len(actual)]
    hw_eval = hw_forecast[:len(actual)]

    mae_lr = mean_absolute_error(actual, lr_eval)
    rmse_lr = np.sqrt(mean_squared_error(actual, lr_eval))

    mae_hw = mean_absolute_error(actual, hw_eval)
    rmse_hw = np.sqrt(mean_squared_error(actual, hw_eval))

    col1, col2 = st.columns(2)

    with col1:
        st.write("Linear Regression")
        st.write(f"MAE: {mae_lr:.2f}")
        st.write(f"RMSE: {rmse_lr:.2f}")

    with col2:
        st.write("Holt-Winters")
        st.write(f"MAE: {mae_hw:.2f}")
        st.write(f"RMSE: {rmse_hw:.2f}")

    # ======================================================
    # BUSINESS INSIGHTS
    # ======================================================

    st.subheader("Business Insights")

    if hw_forecast.mean() > monthly_sales["Amount"].mean():
        st.success(
            "Forecast indicates an upward sales trend. "
            "Inventory expansion and growth planning are recommended."
        )
    else:
        st.warning(
            "Sales trend appears stable or declining. "
            "Marketing and demand optimization strategies are advised."
        )
