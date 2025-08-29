import streamlit as st
import joblib
import numpy as np

st.title("Teen Phone addiction predictor")

model = joblib.load("C:/Learning/project/project for apply/predict phone addiction/save_models/lasso_model.pkl")

st.write("Nhap thong tin de du doan Addiction Level")

daily_usage = st.slider("Daily Usage Hours", 0, 12, 6)
sleep_hours = st.slider("Sleep Hours", 3, 10, 7)
phone_checks = st.slider("Phone Checks Per Day", 10, 200, 80)
apps_used = st.slider("Apps Used Daily", 5, 20, 12)
social_media = st.slider("Time on Social Media (hours)", 0, 5, 2)
gaming = st.slider("Time on Gaming (hours)", 0, 4, 1)

if st.button("Dự đoán"):
    input_features = [daily_usage, sleep_hours, phone_checks, apps_used, social_media, gaming]
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)
    st.success(f"🔮 Addiction Level dự đoán: {prediction[0]:.2f}")


