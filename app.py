import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load model
model = load_model('model/ann_model.h5')

# Dummy label encoders and scaler (harus disamakan dengan saat training)
# Buat ulang dari data yang sama
df = pd.read_csv('data/clothes_price_prediction_data.csv')
X = df.drop('Price', axis=1)

label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

scaler = StandardScaler()
scaler.fit(X)

st.title("Clothes Price Predictor 👕")

# Input form
input_data = {}
for col in X.columns:
    if col in label_encoders:
        unique_vals = df[col].unique().tolist()
        input_data[col] = st.selectbox(col.capitalize(), unique_vals)
    else:
        input_data[col] = st.number_input(f"{col.capitalize()}", value=0.0)

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0][0]
    st.success(f"Predicted Price: ${prediction:.2f}")
