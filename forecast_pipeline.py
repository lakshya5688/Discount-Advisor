import pandas as pd
from prophet import Prophet
import os

def forecast_product_sales(product_id, csv_path="sales_CA1_sample.csv", forecast_days=14):
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
        df = pd.read_csv(csv_path)

        required_columns = ['item_id', 'date', 'sales', 'cat_id']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column.")
        
        product_data = df[df['item_id'] == product_id]
        if product_data.empty:
            raise ValueError(f"Product ID '{product_id}' not found.")

        category = product_data['cat_id'].iloc[0]
        product_df = product_data.sort_values("date")

        # Take last 90 rows or as many as available
        if len(product_df) >= 90:
            product_df = product_df.tail(90)
        elif len(product_df) >= 14:
            pass
        else:
            print(f"⚠️ WARNING: Only {len(product_df)} rows. Forecast may be unreliable.")

        # Preprocess
        product_df['date'] = pd.to_datetime(product_df['date'])
        product_df = product_df.rename(columns={"date": "ds", "sales": "y"})
        product_df = product_df[['ds', 'y']]
        product_df = product_df[product_df['y'] > 0]

        # Outlier removal using IQR
        q1, q3 = product_df['y'].quantile([0.25, 0.75])
        iqr = q3 - q1
        clean_df = product_df[(product_df['y'] >= q1 - 1.5 * iqr) & (product_df['y'] <= q3 + 1.5 * iqr)]
        if len(clean_df) >= 14:
            product_df = clean_df

        # Train Prophet
        model = Prophet(daily_seasonality=True)
        model.fit(product_df)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        forecast_tail = forecast.tail(forecast_days)

        # Compute averages
        recent_mean = product_df.tail(forecast_days)['y'].mean()
        forecast_mean = forecast_tail['yhat'].mean()

        # --- Floating-Point Discount Logic ---
        if forecast_mean < recent_mean and recent_mean > 0:
            decline_ratio = (recent_mean - forecast_mean) / recent_mean
            discount_percent = min(25.0, max(1.0, decline_ratio * 100))
            discount = f"{discount_percent:.1f}%"
            recommendation = "Apply discount"
            trend = "decreasing"
        else:
            discount = "0%"
            recommendation = "No discount needed"
            trend = "increasing"
        # -------------------------------------

        result = {
            "product_id": product_id,
            "cat_id": category,
            "forecast": forecast_tail[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'predicted_sales'}).to_dict(orient='records'),
            "trend": trend,
            "recommendation": recommendation,
            "suggested_discount": discount
        }
        return result

    except Exception as e:
        return {"error": str(e)}


def get_dropdown_options(csv_path="sales_CA1_sample.csv"):
    df = pd.read_csv(csv_path)
    df = df[['item_id', 'cat_id']].drop_duplicates()
    return df
