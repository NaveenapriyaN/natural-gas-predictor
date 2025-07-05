import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
import streamlit as st
import math

# -------------------- Data Preparation -------------------- #
df = pd.read_csv("Nat_Gas.csv")
df.rename(columns={'Dates': 'Date', 'Prices': 'Price'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['TimeIndex'] = np.arange(len(df))
df['Month'] = df['Date'].dt.month

X = df[['TimeIndex', 'Month']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

# Price estimator for any date
def estimate_price(input_date_obj):
    months_since_start = (input_date_obj.year - df['Date'].iloc[0].year) * 12 + (input_date_obj.month - df['Date'].iloc[0].month)
    input_time_index = df['TimeIndex'].iloc[0] + months_since_start
    input_month = input_date_obj.month
    predicted_price = model.predict([[input_time_index, input_month]])[0]
    return round(predicted_price, 2)

# -------------------- Streamlit App -------------------- #
st.title("🔮 Natural Gas Price & Contract Simulator")

tabs = st.tabs(["📈 Price Forecast", "💼 Contract Estimator"])

# -------------------- Tab 1: Price Forecast -------------------- #
with tabs[0]:
    st.header("📈 Historical and Future Natural Gas Prices")
    st.line_chart(df.set_index('Date')['Price'])

    input_date = st.date_input("Pick a future date to predict price:")
    if input_date:
        predicted = estimate_price(input_date)
        st.success(f"📅 Estimated Price on {input_date.strftime('%Y-%m-%d')}: **${predicted}**")

    # 12-month forecast
    future_months = 12
    last_time_index = df['TimeIndex'].iloc[-1]
    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.DateOffset(months=1), periods=future_months, freq='M')
    future_df = pd.DataFrame({
        'Date': future_dates,
        'TimeIndex': np.arange(last_time_index + 1, last_time_index + 1 + future_months),
        'Month': future_dates.month
    })
    future_df['PredictedPrice'] = model.predict(future_df[['TimeIndex', 'Month']])

    st.write("### 🔮 1-Year Forecast")
    st.line_chart(future_df.set_index('Date')['PredictedPrice'])

# -------------------- Tab 2: Contract Valuation -------------------- #
with tabs[1]:
    st.header("💼 Price a Commodity Storage Contract")
    st.write("Estimate profit or loss from storing and trading natural gas.")

    with st.expander("📥 Injection Schedule"):
        in_dates = st.text_area("Injection Dates (YYYY-MM-DD, comma-separated)", "2025-07-01, 2025-08-01")
        in_vols = st.text_area("Injection Volumes (same order, comma-separated)", "100000, 100000")

    with st.expander("📤 Withdrawal Schedule"):
        out_dates = st.text_area("Withdrawal Dates (YYYY-MM-DD, comma-separated)", "2025-11-01, 2025-12-01")
        out_vols = st.text_area("Withdrawal Volumes (same order, comma-separated)", "100000, 100000")

    st.write("---")
    rate = st.number_input("Rate limit (units/day)", 100000)
    max_storage = st.number_input("Maximum Storage Capacity (units)", 500000)
    storage_cost = st.number_input("Monthly Storage Fee ($)", 10000)
    inj_wdr_cost_rate = st.number_input("Injection/Withdrawal Cost Rate ($ per unit)", 0.0005, format="%.4f")

    if st.button("🔍 Calculate Contract Value"):
        try:
            # Parse input
            in_dates = [datetime.strptime(d.strip(), "%Y-%m-%d").date() for d in in_dates.split(",")]
            in_vols = [float(v.strip()) for v in in_vols.split(",")]
            out_dates = [datetime.strptime(d.strip(), "%Y-%m-%d").date() for d in out_dates.split(",")]
            out_vols = [float(v.strip()) for v in out_vols.split(",")]

            volume = 0
            buy_cost = 0
            cash_in = 0
            all_dates = sorted(set(in_dates + out_dates))

            for i in range(len(all_dates)):
                day = all_dates[i]

                if day in in_dates:
                    vol = in_vols[in_dates.index(day)]
                    if volume + vol > max_storage:
                        st.warning(f"❌ Cannot inject on {day} – not enough storage.")
                        continue
                    price = estimate_price(day)
                    buy_cost += vol * price + vol * inj_wdr_cost_rate
                    volume += vol
                elif day in out_dates:
                    vol = out_vols[out_dates.index(day)]
                    if vol > volume:
                        st.warning(f"❌ Cannot withdraw on {day} – not enough gas stored.")
                        continue
                    price = estimate_price(day)
                    cash_in += vol * price - vol * inj_wdr_cost_rate
                    volume -= vol

            storage_months = (max(out_dates).year - min(in_dates).year) * 12 + (max(out_dates).month - min(in_dates).month)
            total_storage_cost = storage_months * storage_cost
            net_value = cash_in - buy_cost - total_storage_cost

            st.success(f"💰 Contract Value: **${net_value:,.2f}**")
            st.info(f"Buy Cost: ${buy_cost:,.2f} | Sell Revenue: ${cash_in:,.2f} | Storage Fees: ${total_storage_cost:,.2f}")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
