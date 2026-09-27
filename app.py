"""
Streamlit app for the demand forecasting model.
Run locally with:  streamlit run app.py
Deploy free at: https://share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail Demand Forecast", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("outputs/xgb_forecast_model.pkl")
    feature_cols = joblib.load("outputs/feature_columns.pkl")
    history = joblib.load("outputs/recent_history.pkl")
    return model, feature_cols, history

model, feature_cols, history = load_artifacts()
explainer = shap.TreeExplainer(model)

st.title("🛒 Retail Demand Forecast")
st.write(
    "Forecast weekly sales for a store/department using external "
    "regressors (temperature, holidays, promotions)."
)

tab1, tab2 = st.tabs(["Batch Forecast (CSV)", "Single Forecast"])

# ---------------- TAB 1: Batch ----------------
with tab1:
    st.write(
        "Upload a CSV with columns: " + ", ".join(feature_cols) +
        " (lag_1, lag_52, rolling_mean_4 can be looked up automatically "
        "if you instead provide Store, Dept, Date — see Single Forecast "
        "tab for that simpler flow)."
    )
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing = set(feature_cols) - set(data.columns)
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            preds = model.predict(data[feature_cols])
            result = data.copy()
            result["predicted_weekly_sales"] = preds
            st.dataframe(result.head(50))
            st.download_button(
                "Download forecasts",
                result.to_csv(index=False),
                file_name="forecasts.csv",
            )

# ---------------- TAB 2: Single forecast with SHAP ----------------
with tab2:
    st.write("Pick a store/department that exists in recent history, then adjust regressors to forecast next week's sales.")
    col1, col2 = st.columns(2)

    store = col1.selectbox("Store", sorted(history["Store"].unique()))
    dept_options = sorted(history[history["Store"] == store]["Dept"].unique())
    dept = col2.selectbox("Dept", dept_options)

    recent = history[(history["Store"] == store) & (history["Dept"] == dept)].sort_values("Date")

    if recent.empty:
        st.warning("No recent history for this Store/Dept combination.")
    else:
        last_row = recent.iloc[-1]

        col1, col2, col3 = st.columns(3)
        temperature = col1.number_input("Temperature (°F)", value=float(last_row["Temperature"]))
        fuel_price = col2.number_input("Fuel Price", value=float(last_row["Fuel_Price"]))
        markdown_total = col3.number_input("Total Promotion / MarkDown $", value=float(last_row["MarkDown_total"]))
        is_holiday = col1.selectbox("Is Holiday Week?", [0, 1], index=int(last_row["IsHoliday"]))
        cpi = col2.number_input("CPI", value=float(last_row["CPI"]))
        unemployment = col3.number_input("Unemployment Rate", value=float(last_row["Unemployment"]))

        if st.button("Forecast next week's sales"):
            input_row = {
                "Store": store, "Dept": dept, "IsHoliday": is_holiday,
                "Temperature": temperature, "Fuel_Price": fuel_price,
                "MarkDown_total": markdown_total, "CPI": cpi,
                "Unemployment": unemployment, "Size": last_row["Size"],
                "Type": last_row["Type"], "Year": last_row["Year"],
                "Month": last_row["Month"], "WeekOfYear": last_row["WeekOfYear"],
                "Quarter": last_row["Quarter"], "lag_1": last_row["Weekly_Sales"],
                "lag_52": last_row["lag_52"], "rolling_mean_4": last_row["rolling_mean_4"],
            }
            X_single = pd.DataFrame([input_row])[feature_cols]
            pred = model.predict(X_single)[0]

            st.metric("Forecasted Weekly Sales", f"${pred:,.0f}")

            shap_values = explainer.shap_values(X_single)
            fig, ax = plt.subplots()
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X_single.iloc[0],
                    feature_names=feature_cols,
                ),
                show=False,
            )
            st.pyplot(fig)
            st.caption("Bars show which factors pushed this forecast up (red) or down (blue).")

        st.line_chart(recent.set_index("Date")["Weekly_Sales"], height=200)
        st.caption("Recent sales history for this store/department")

st.divider()
st.caption(
    "Model: XGBoost trained on the Walmart Store Sales Forecasting dataset, "
    "using lag features, temperature, holidays and promotions as regressors."
)
