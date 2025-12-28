import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("🚗 Car Price Prediction")

model = joblib.load("car_price_predict.pkl")

# -------- Inputs --------
car = st.text_input("Car Name")

year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
present = st.number_input("Present Price (Lakhs)", min_value=0.0, value=5.0)
kms = st.number_input("Kms Driven", min_value=0, value=20000)

fuel = st.selectbox("Fuel Type", ["Petrol","Diesel","CNG"])
seller = st.selectbox("Seller Type", ["Dealer","Individual"])
trans = st.selectbox("Transmission", ["Manual","Automatic"])

owner = st.selectbox("Owner", [0,1,2,3]
)

# ---------- Encoding (IMPORTANT) ----------
fuel_map = {"Petrol":0, "Diesel":1, "CNG":2}
seller_map = {"Dealer":0, "Individual":1}
trans_map = {"Manual":0, "Automatic":1}

if st.button("Predict"):
    # Make numpy array (1 row × features)
    x = np.array([
        year,
        present,
        kms,
        fuel_map[fuel],
        seller_map[seller],
        trans_map[trans],
        owner
    ]).reshape(1, -1)

    price = model.predict(x)[0]

    st.success(f"Estimated Selling Price: ₹ {round(price,2)} Lakhs")
