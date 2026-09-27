# Retail Demand Forecasting with External Regressors

## Business problem
Retailers need to forecast weekly sales per store/department to plan
inventory and staffing. Plain historical-sales-only forecasting misses
predictable swings driven by holidays, promotions, and even weather —
this project adds those external regressors to a proper time-series
pipeline.

## Data
[Kaggle: Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)
— weekly sales per store/department, plus external features: Temperature,
Fuel Price, MarkDown1-5 (promotions), CPI, Unemployment, and IsHoliday.

## Approach
1. **Merged** three files (train, features, stores) into one modeling table.
2. **EDA**: confirmed a holiday sales bump, examined temperature/sales
   relationship, found MarkDown columns were mostly NaN because
   promotions only started partway through the data (filled with 0,
   not mean — NaN means "no promo," not "missing value").
3. **Feature engineering**: date parts (year/month/week/quarter),
   `lag_1` (last week), `lag_52` (same week last year — captures yearly
   seasonality), 4-week rolling mean, total promotion spend.
4. **Chronological train/test split** (not random) — a random split
   would let the model "see the future," which doesn't reflect how
   it'll actually be used.
5. **Naive baseline first**: "same week last year" (`lag_52`) as the
   zero-effort benchmark every real model must beat.
6. **Models compared**: Linear Regression, Random Forest, XGBoost.
7. **Weighted MAE**: reported alongside plain MAE/RMSE, weighting
   holiday weeks 5x — matching the original competition's own metric,
   since holiday-week errors cost the business more.
8. **Explainability**: SHAP values show which regressors (holiday,
   markdown, temperature, lag features) drove each forecast.
9. **Deployment**: Streamlit app for batch CSV forecasting and a
   single store/department forecast with adjustable regressors and a
   live SHAP explanation.

## Results

| Model | MAE | RMSE |
|---|---|---|
| Naive baseline (same week last year) | 1754.75 | 3745.99 |
| Linear Regression | 1486.82 | 2954.49 |
| Random Forest | 1236.17 | 2630.08 |
| XGBoost | 1263.76 | 2675.23 |

**Weighted MAE (holiday weeks weighted 5x), XGBoost:** 1312.52 — higher
than XGBoost's plain MAE (1263.76), confirming the model's errors are
somewhat larger on the highest-stakes holiday weeks, where accuracy
matters most to the business.

**Key finding:** all three models beat the naive "same week last year"
baseline by 15-30% on MAE, confirming the added regressors and lag
features carry real signal. Random Forest slightly outperformed
XGBoost on both MAE and RMSE despite XGBoost being the more commonly
assumed default — a reminder to actually compare models rather than
assume the "fancier" one wins. Given more time, Random Forest would be
the one to tune further and also evaluate with the weighted-MAE metric.

**EDA finding worth calling out:** holiday weeks average ~$17,036 in
sales vs. ~$15,901 on regular weeks (~7% lift), consistent with the
lag/holiday features carrying predictive value.

## What I'd improve with more time
- Store-level or department-level models instead of one global model
  (sales patterns likely differ a lot by store type/size)
- Proper time-series cross-validation (rolling-origin) instead of one
  fixed cutoff date
- External holiday calendar with named holidays (Thanksgiving vs.
  Labor Day likely have very different effects, currently collapsed
  into one IsHoliday flag)

## How to run
```bash
pip install -r requirements.txt
mkdir -p data outputs
# place train.csv, features.csv, stores.csv inside data/
python eda_and_modeling.py
streamlit run app.py
```

## Live demo
https://fraud-detection-app-8xu4xttzdyy4xuaumsvnmw.streamlit.app/
