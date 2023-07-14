#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import streamlit as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Load the dataset
data=pd.read_excel("bankruptcy-prevention.xlsx")

# Encoding Data
label_encoder = preprocessing.LabelEncoder()
data[' class'] = label_encoder.fit_transform(data[' class'])


# Split the data into features (X) and target (y)
x = data.drop(' class', axis=1)
y = data[' class']
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Create the AdaBoost classifier
kfold=KFold(n_splits=5,random_state=72,shuffle=True)
model_ab= AdaBoostClassifier(n_estimators=60, random_state=8)        
result_ab = cross_val_score(model_ab, x, y, cv=kfold)
#Accuracy
print(result_ab.mean())


filename = 'final_Adaboost_model.pkl'
pickle.dump(model_ab, open(filename,'wb'))
model_ab.fit(x,y)
pk=model_ab.predict(x_test)

st.title("Bankruptcy-Prevention")

risk_mapping = {
    'Low': 0,
    'Medium': 0.5,
    'High': 1
}

# Now you can use the assigned numerical values in your predictions or further processing

Industrial_risk = st.selectbox('Industrial_risk', ('Low','Medium','High'))
Management_risk = st.selectbox(' Management_risk', ('Low','Medium','High'))
Financial_flexibility = st.selectbox(' Financial_flexibility',('Low','Medium','High'))
Credibility = st.selectbox(' Credibility',('Low','Medium','High'))
Competitiveness = st.selectbox(' Competitiveness',('Low','Medium','High'))
Operating_risk = st.selectbox(' Operating_risk', ('Low','Medium','High'))


if st.button('Prevention Type'):
    df = {
        'industrial_risk': risk_mapping[Industrial_risk],
        ' management_risk': risk_mapping[Management_risk],
        ' financial_flexibility': risk_mapping[Financial_flexibility],
        ' credibility': risk_mapping[Credibility],
        ' competitiveness': risk_mapping[Competitiveness],
        ' operating_risk': risk_mapping[Operating_risk]
    }

    df1 = pd.DataFrame(df,index=[1])
    predictions = model_ab.predict(df1)

    if predictions.any() == 1:
        prediction_value = 'Non-Bankruptcy'
    else:
        prediction_value = 'Bankruptcy'
    
    st.title("Business type is " + str(prediction_value))


    





