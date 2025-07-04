import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import streamlit as st

# Load the CSV file
df = pd.read_csv("Nat_Gas.csv")  # your actual file name

# Rename columns to standard ones if needed
df.rename(columns={'Dates': 'Date', 'Prices': 'Price'}, inplace=True)

# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Add time-based features
df['TimeIndex'] = np.arange(len(df))
df['Month'] = df['Date'].dt.month

# Features and target
X = df[['TimeIndex', 'Month']]
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("📊 Natural Gas Forecasting App (ML Powered)")

st.write("### 📈 Historical Natural Gas Prices")
st.line_chart(df.set_index('Date')['Price'])

# User input
input_date = st.date_input("Pick a future date to predict price:")

# Prediction logic
def estimate_price(input_date_obj):
    months_since_start = (input_date_obj.year - df['Date'].iloc[0].year) * 12 + (input_date_obj.month - df['Date'].iloc[0].month)
    input_time_index = df['TimeIndex'].iloc[0] + months_since_start
    input_month = input_date_obj.month
    predicted_price = model.predict([[input_time_index, input_month]])[0]
    return round(predicted_price, 2)

if input_date:
    predicted = estimate_price(input_date)
    st.success(f"📅 Estimated Price on {input_date.strftime('%Y-%m-%d')}: **${predicted}**")

# Future forecast (12 months)
future_months = 12
last_time_index = df['TimeIndex'].iloc[-1]
future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.DateOffset(months=1), periods=future_months, freq='M')

future_df = pd.DataFrame({
    'Date': future_dates,
    'TimeIndex': np.arange(last_time_index + 1, last_time_index + 1 + future_months),
    'Month': future_dates.month
})
future_df['PredictedPrice'] = model.predict(future_df[['TimeIndex', 'Month']])

st.write("### 🔮 1-Year Price Forecast")
st.line_chart(future_df.set_index('Date')['PredictedPrice'])
