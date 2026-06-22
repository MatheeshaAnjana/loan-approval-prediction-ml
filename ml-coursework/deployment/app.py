from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Column order MUST match the order used when the scaler/model were trained
FEATURE_COLUMNS = [
    'CoapplicantIncome',
    'TotalIncome',
    'ApplicantIncome',
    'LoanAmount',
    'Credit_History',
    'Loan_Amount_Term',
    'Education',
    'Married'
]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, form_data=None)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    try:
        married = float(form.get("Married"))
        education = float(form.get("Education"))
        applicant_income = float(form.get("ApplicantIncome"))
        coapplicant_income = float(form.get("CoapplicantIncome"))
        loan_amount = float(form.get("LoanAmount"))
        loan_amount_term = float(form.get("Loan_Amount_Term"))
        credit_history = float(form.get("Credit_History"))

        total_income = applicant_income + coapplicant_income

        sample = pd.DataFrame([[
            coapplicant_income,
            total_income,
            applicant_income,
            loan_amount,
            credit_history,
            loan_amount_term,
            education,
            married
        ]], columns=FEATURE_COLUMNS)

        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)

        approved = bool(prediction[0] == 1)
        result = {
            "approved": approved,
            "message": "Loan Approved" if approved else "Loan Rejected"
        }

    except (TypeError, ValueError):
        result = {
            "approved": None,
            "message": "Please fill in all fields with valid values."
        }

    return render_template("index.html", result=result, form_data=form)


if __name__ == "__main__":
    app.run(debug=True)