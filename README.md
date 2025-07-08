📊 Demand Forecasting & Discount Recommendation System
An intelligent product-level demand forecasting tool built using Facebook Prophet, designed to analyze recent sales trends and recommend category-aware discounts. This tool is ideal for retail teams to dynamically assess demand and apply smart pricing strategies.

🔧 Key Features
Dynamic Forecasting: Generates 14-day forecasts using Prophet on-demand for any selected item_id

Storage-Efficient: Loads and processes only the latest 90 rows per product (or fewer if data is limited)

Robust Preprocessing: Includes outlier removal (IQR), negative sales filtering, and flexible data fallback

Category-Aware Discounts: Suggests discount percentages based on both the forecast trend and product category (cat_id)

Interactive Dashboard: Streamlit UI with:

Category filter (cat_id)

Product dropdown auto-filtered by category

Trend summary and discount suggestion

Plotly-based interactive forecast visualization

🛠 Tech Stack
Forecasting: Facebook Prophet

Visualization: Plotly, Seaborn, Matplotlib

Frontend: Streamlit

Data Handling: Pandas, NumPy

Planned Backend: MERN Stack (MongoDB, Express, React, Node.js)

