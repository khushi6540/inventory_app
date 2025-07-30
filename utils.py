# utils.py
import pandas as pd
import joblib

def load_model():
    model = joblib.load('xgb_demand_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

def preprocess_input(df):
    df['Date'] = pd.to_datetime(df['Date'],format='%d-%m-%Y')
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    df = df.drop(['Date', 'Store ID', 'Product ID'], axis=1)
    df = pd.get_dummies(df, drop_first=True)

    return df
