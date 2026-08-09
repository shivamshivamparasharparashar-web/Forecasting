# Financial Prediction Arena

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `simulated_financial_forecasting_data.csv` in the same directory as `app.py`,
or upload the CSV from the Streamlit sidebar.

## Important modeling note

The supplied dataset has no date/time column. The app therefore performs conditional
regression for `target_sales`; it does not perform a genuine future time-series forecast.
