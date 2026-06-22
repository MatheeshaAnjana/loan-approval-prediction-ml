from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_COLUMNS = [
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
]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, form_data=None)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    try:
        gender = float(form.get("Gender"))
        married = float(form.get("Married"))
        dependents = float(form.get("Dependents"))
        education = float(form.get("Education"))
        self_employed = float(form.get("Self_Employed"))
        applicant_income = float(form.get("ApplicantIncome"))
        coapplicant_income = float(form.get("CoapplicantIncome"))
        loan_amount = float(form.get("LoanAmount"))
        loan_amount_term = float(form.get("Loan_Amount_Term"))
        credit_history = float(form.get("Credit_History"))
        property_area = float(form.get("Property_Area"))

        total_income = applicant_income + coapplicant_income

        sample = pd.DataFrame([[
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            property_area,
            total_income
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