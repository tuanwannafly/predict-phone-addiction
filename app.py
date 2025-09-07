import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "save_models", "lasso_model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "teen_phone_addiction_dataset.csv")

model = joblib.load(MODEL_PATH)
data = pd.read_csv(CSV_PATH)
feature_names = data.select_dtypes(include="number").drop("Addiction_Level", axis=1).columns

st.write("Nhap thong tin de du doan Addiction Level")


inputs = {}


inputs["Daily_Usage_Hours"] = st.number_input("Daily Usage Hours", 0.0, 24.0, 5.0, step=0.5)
inputs["Sleep_Hours"] = st.number_input("Sleep Hours", 0.0, 24.0, 7.0, step=0.5)
inputs["Academic_Performance"] = st.slider("Academic Performance", 0, 100, 75)
inputs["Social_Interactions"] = st.slider("Social Interactions", 0, 10, 5)
inputs["Exercise_Hours"] = st.number_input("Exercise Hours", 0.0, 10.0, 1.0, step=0.5)
inputs["Anxiety_Level"] = st.slider("Anxiety Level", 0, 10, 5)
inputs["Depression_Level"] = st.slider("Depression Level", 0, 10, 5)
inputs["Self_Esteem"] = st.slider("Self Esteem", 0, 10, 7)
inputs["Parental_Control"] = st.slider("Parental Control", 0, 10, 0)
inputs["Screen_Time_Before_Bed"] = st.number_input("Screen Time Before Bed (hours)", 0.0, 10.0, 1.0, step=0.5)
inputs["Phone_Checks_Per_Day"] = st.number_input("Phone Checks Per Day", 0, 500, 100, step=5)
inputs["Apps_Used_Daily"] = st.number_input("Apps Used Daily", 0, 100, 15, step=1)
inputs["Time_on_Social_Media"] = st.number_input("Time on Social Media (hours)", 0.0, 24.0, 3.0, step=0.5)
inputs["Time_on_Gaming"] = st.number_input("Time on Gaming (hours)", 0.0, 24.0, 2.0, step=0.5)
inputs["Time_on_Education"] = st.number_input("Time on Education (hours)", 0.0, 24.0, 1.0, step=0.5)
inputs["Family_Communication"] = st.slider("Family Communication", 0, 10, 4)
inputs["Weekend_Usage_Hours"] = st.number_input("Weekend Usage Hours", 0.0, 24.0, 7.0, step=0.5)


if st.button("🔮 Dự đoán"):
    input_df = pd.DataFrame([inputs])
    # Đảm bảo đúng thứ tự cột
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(input_df)[0]
    st.success(f"🔮 Addiction Level dự đoán: {prediction:.2f}")

