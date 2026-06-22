import joblib
import numpy as np
import pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

print("\n===== Loan Approval Predictor =====\n")

Gender = float(input("Gender (Male=1, Female=0): "))
Married = float(input("Married (Yes=1, No=0): "))
Dependents = float(input("Dependents (0,1,2,3): "))
Education = float(input("Education (Graduate=1, Not Graduate=0): "))
Self_Employed = float(input("Self Employed (Yes=1, No=0): "))
ApplicantIncome = float(input("Applicant Income: "))
CoapplicantIncome = float(input("Coapplicant Income: "))
LoanAmount = float(input("Loan Amount: "))
Loan_Amount_Term = float(input("Loan Amount Term: "))
Credit_History = float(input("Credit History (1=Good, 0=Bad): "))
Property_Area = float(input("Property Area (Urban=2, Semiurban=1, Rural=0): "))

TotalIncome = ApplicantIncome + CoapplicantIncome

sample = pd.DataFrame([[
    Gender,
    Married,
    Dependents,
    Education,
    Self_Employed,
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area,
    TotalIncome
]], columns=[
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
    'Property_Area',
    'TotalIncome'
])

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

if prediction[0] == 1:
    print("\n✅ Loan Approved")
else:
    print("\n❌ Loan Rejected")