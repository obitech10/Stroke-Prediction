# linear algebra
import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

# For designing application interface
import streamlit as st

# SMOTE
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

# data split
from sklearn.model_selection import train_test_split

# scaling
from sklearn.preprocessing import StandardScaler

# model Evaluation
from sklearn import metrics

# model development
from sklearn.ensemble import RandomForestClassifier

# for model accuracy
from sklearn.metrics import make_scorer, accuracy_score


st.write("""
# STROKE PREDICTION APP
""")
st.write("---")

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

df.info()

# drop the ID column
df = df.drop('id', axis=1)

# MISSING DATA
df['bmi'].fillna((df['bmi'].mean()), inplace=True)
df.info()

st.sidebar.header("Specify User Input Features")


def user_input_features():
    gender = st.sidebar.selectbox("Gender:", ("Male", "Female", "Other"))
    age = st.sidebar.slider("Patient age:", 0, 120, 60)
    hypertension = st.sidebar.selectbox("Has Patient had Hypertension:", ("Yes", "No"))
    heart_disease = st.sidebar.selectbox("Does Patient Have/Has ever had a Heart Disease:", ("Yes", "No"))
    ever_married = st.sidebar.selectbox("Ever Married:", ("Yes", "No"))
    work_type = st.sidebar.selectbox("Work Type:", ("children", "Govt_job", "Never_worked", "Private", "Self-employed"))
    Residence_type = st.sidebar.selectbox("Residence Type:", ("Rural", "Urban"))
    avg_glucose_level = st.sidebar.number_input("Enter Patient Average Glucose level:")
    bmi = st.sidebar.number_input("Enter Patient BMI:")
    smoking_status = st.sidebar.selectbox("Smoking Status:", ("formerly smoked", "never smoked", "smokes", "Unknown"))

    data = {'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status}

    features = pd.DataFrame(data, index=[0])
    return features


data = user_input_features()

st.header("Specified User Input")
st.write(data)
st.write("---")

y = df['stroke']
# drop the Stroke column
df = df.drop(columns=['stroke'])
df = pd.concat([data, df], axis=0)

encode = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'heart_disease', 'hypertension']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

# select the first row(User Input)
data = df[:1]
df = df[1:]

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30, random_state=1)

# SMOTE oversampling

SMOTE_oversample = SMOTE(random_state=1)
X_train, y_train = SMOTE_oversample.fit_resample(X_train, y_train.ravel())

# MODELLING
# create object model
RF_model = RandomForestClassifier(n_estimators=400, min_samples_split=100, min_samples_leaf=30, max_features=6, max_depth=11, bootstrap=False)

# fit the model
RF_model.fit(X_train, y_train)

# MODEL APPLICATION
prediction = RF_model.predict(data)
probability = np.round(RF_model.predict_proba(data)*100)

if prediction == 1:
    prediction = "Patient will have a stroke"
else:
    prediction = "Patient will not have a stroke"

st.header("Prediction")
st.write(prediction)
st.write("---")

st.header("Prediction Probability")
st.write('0 - Patient will not have stroke')
st.write('1 - Patient will have stroke')
st.write(probability)
st.write("---")

percent = RF_model.predict(X_test)
accu = np.round(accuracy_score(y_test, percent)*100)

st.write("Prediction Accuracy= ", accu, "%")
st.write("---")
