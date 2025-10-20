import streamlit as st
import pandas as pd
import joblib

model = joblib.load("svm_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.image("https://pngimg.com/uploads/heart/heart_PNG691.png", width=200)
st.title("Heart stroke prediction by Gaurav Pandey")
st.markdown("provide the followeing details")

# Input widgets
age = st.slider('Age', 18, 40)
sex = st.selectbox('Sex', ['M', 'F'])
chest_pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
fasting_bs = st.selectbox('Fastin Blood Sugar>120 mg/dl', [0, 1])
cholesterol = st.number_input('Cholesterol (mg/dl)', 100, 600, 200)
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
exercise_engine = st.selectbox('Exercise Inducted Angine', ['Y', 'N'])
oldpeak = st.slider('oldpeak (ST Depression)', 0.0, 6.0, 1.0)
st_slope = st.selectbox('ST_Slope', ['Up', 'Flat', 'Down'])


# Prediction button and logic
if st.button("Predict"):
    # Prepare raw input data
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngine_' + exercise_engine: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create initial DataFrame
    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns are present (one-hot encoding)
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns and select only expected ones
    input_df = input_df[expected_columns]
    
    # Scale the input data
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Display result
    if prediction == 1:
        st.error("High Risk Of Heart Disease")
    else:
        st.success("Low Risk Of Heart Disease")
    