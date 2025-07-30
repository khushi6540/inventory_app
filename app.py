# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
    fig1 = go.Figure()
    for status in df["Stock Status"].unique():
        fig1.add_trace(go.Histogram(
            x=df[df["Stock Status"] == status]["Stock Status"],
            name=status
        ))
    fig1.update_layout(title="Inventory Risk Status", barmode='stack')
    st.plotly_chart(fig1, use_container_width=True)
    
    
    fig2 = go.Figure()
    for status in df["Stock Status"].unique():
        filtered = df[df["Stock Status"] == status]
        fig2.add_trace(go.Bar(
            x=filtered["Product ID"],
            y=filtered["Overstock"],
            name=status
        ))
    fig2.update_layout(title="Product-wise Overstock Analysis")
    st.plotly_chart(fig2, use_container_width=True)
    
       # ✅ Visualization 3: Bar Plot of Predicted Demand by Category
        
    
    # Create bar plot
    category_demand = df.groupby("Category")["Predicted Demand"].sum().reset_index()
    
    fig3 = go.Figure(go.Bar(
        x=category_demand["Category"],
        y=category_demand["Predicted Demand"],
        marker_color="indianred"
    ))
    fig3.update_layout(title="Total Predicted Demand by Category")
    st.plotly_chart(fig3, use_container_width=True) 
