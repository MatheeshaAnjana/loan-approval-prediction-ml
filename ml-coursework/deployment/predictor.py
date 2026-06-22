import joblib
import pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

print("\n===== Loan Approval Predictor =====\n")

CoapplicantIncome = float(input("Coapplicant Income: "))
ApplicantIncome = float(input("Applicant Income: "))
LoanAmount = float(input("Loan Amount: "))
Credit_History = float(input("Credit History (1=Good, 0=Bad): "))
Loan_Amount_Term = float(input("Loan Amount Term: "))
Education = float(input("Education (Graduate=1, Not Graduate=0): "))
Married = float(input("Married (Yes=1, No=0): "))

TotalIncome = ApplicantIncome + CoapplicantIncome

# Column order MUST match the order used when the scaler/model were trained:
# ['CoapplicantIncome', 'TotalIncome', 'ApplicantIncome', 'LoanAmount',
#  'Credit_History', 'Loan_Amount_Term', 'Education', 'Married']
sample = pd.DataFrame([[
    CoapplicantIncome,
    TotalIncome,
    ApplicantIncome,
    LoanAmount,
    Credit_History,
    Loan_Amount_Term,
    Education,
    Married
]], columns=[
    'CoapplicantIncome',
    'TotalIncome',
    'ApplicantIncome',
    'LoanAmount',
    'Credit_History',
    'Loan_Amount_Term',
    'Education',
    'Married'
])

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

if prediction[0] == 1:
    print("\n✅ Loan Approved")
else:
    print("\n❌ Loan Rejected")