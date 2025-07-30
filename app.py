# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_model, preprocess_input

st.set_page_config(page_title="Inventory Management Dashboard", layout="wide")

st.title(" Inventory Management System")

uploaded_file = st.file_uploader("Upload Inventory Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Raw Dataset")
    st.dataframe(df.head())

    model, model_features = load_model()
    df_processed = preprocess_input(df)

    # Align with trained model features
    for col in model_features:
        if col not in df_processed.columns:
            df_processed[col] = 0
    df_processed = df_processed[model_features]

    df["Predicted Demand"] = model.predict(df_processed)
    df["Overstock"] = df["Inventory Level"] - df["Predicted Demand"]
    df["Stock Status"] = df["Overstock"].apply(lambda x: "Overstock" if x > 50 else ("Stockout Risk" if x < 0 else "Optimal"))

    st.subheader("📈 Inventory Insights")
    fig1 = px.histogram(df, x="Stock Status", color="Stock Status", title="Inventory Risk Status")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(df, x="Product ID", y="Overstock", color="Stock Status", title="Product-wise Overstock Analysis")
    st.plotly_chart(fig2, use_container_width=True)

   # ✅ Visualization 3: Bar Plot of Predicted Demand by Category
    category_demand = df.groupby("Category")["Predicted Demand"].sum().reset_index()

# Create bar plot
    fig3 = px.bar(category_demand, x="Category", y="Predicted Demand", title="Total Predicted Demand by Category", color="Category")
    st.plotly_chart(fig3, use_container_width=True)


    st.subheader("📊 Demand Prediction Table")
    st.dataframe(df[["Store ID", "Product ID", "Category", "Region", "Inventory Level", "Predicted Demand", "Stock Status"]])
