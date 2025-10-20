import streamlit as st
import pandas as pd
import joblib

model=joblib.load('svm_heart.pkl')
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load('columns.pkl')

st.title("Heart stroke prediction by Gaurav Pandey")
st.markdown('provide the folloewing details')
age=st.slider('age',18,100,40)
sex=st.selectbox('sex',['M','F'])
chest_pain=st.selectbox("Chest Pain Type",['ATA','NAP','TA','ASY'])
resting_bp=st.number_input("Resting Blood Pressure(mm Hg)",80,200,120)
fasting_bs=st.selectbox("Fastin Blood Sugar>120 mg/dl",[0,1])
cholesterol=st.number_input('cholestrol(mg/dl)',100,600,200)
resting_ecg=st.selectbox('Resting ECG',['Normal',"ST","LVH"])
max_hr=st.slider("Max Heart Rate",60,220,150)
exercise_engine=st.selectbox("Exercise_Inducted Angine",['y','N'])
oldpeak =st.slider('oldpeak (ST Depression)',0.0,6.0,1.0)
st_slope=st.selectbox("ST Slope",['Up','Flat','Down'])

if st.button("Predict"):
    raw_input={
        "Age":age,
        "RestingBP":resting_bp,
        "Cholestrol":cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR':max_hr,
        'Sex_'+sex: 1,
        'ChestPainType_'+chest_pain:1,
        'RestingECG_'+resting_ecg:1,
        'ExerciseAngine_'+exercise_engine:1,
        'ST_Slope_'+st_slope:1

}
input_df=pd.DataFrame([raw_input])
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col]=0
input_df=input_df[expected_columns]
scaled_input=scaler.transform(input_df)
prediction=model.predict(scaled_input)[0]

if prediction==1:
    st.error("High Risk Of Heart Disease")
else:
    st.success("Low Risk Of Heart Disease")
   