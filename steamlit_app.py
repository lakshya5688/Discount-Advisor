import streamlit as st
from forecast_pipeline import forecast_product_sales, get_dropdown_options
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Demand Forecast Assistant", layout="centered")
st.title("Product Demand Forecaster")

# Load options
df_options = get_dropdown_options()
categories = df_options['cat_id'].unique()

# Step 1: Select Category
selected_cat = st.selectbox("Select Category", sorted(categories))

# Step 2: Filter items for selected category
filtered_df = df_options[df_options['cat_id'] == selected_cat]
item_ids = filtered_df['item_id'].unique()
selected_item = st.selectbox("Select Product ID", sorted(item_ids))

# Button to forecast
if st.button("Generate Forecast"):
    with st.spinner("Running Prophet forecast..."):
        result = forecast_product_sales(selected_item)

    if "error" in result:
        st.error(f"{result['error']}")
    else:
        st.success("Forecast generated!")

        # Metadata
        st.markdown(f"**Category**: {result['cat_id']}")
        st.markdown(f"**Trend**: `{result['trend']}`")
        st.markdown(f"**Recommendation**: :blue[{result['recommendation']}]")
        st.markdown(f"**Suggested Discount**: :green[{result['suggested_discount']}]")

        # Forecast table
        st.subheader("Next 14 Days Forecast")
        forecast_df = pd.DataFrame(result['forecast'])
        st.dataframe(forecast_df, use_container_width=True)

        # Interactive Plot (Plotly)
        st.subheader("Forecast Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted_sales'],
            mode='lines+markers',
            name='Forecasted Sales',
            line=dict(color='royalblue', width=3)
        ))
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Predicted Sales',
            margin=dict(l=40, r=40, t=40, b=40),
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
