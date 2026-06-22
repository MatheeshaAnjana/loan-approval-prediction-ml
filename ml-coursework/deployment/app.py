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
    applicant_name = form.get("ApplicantName", "").strip()

    try:
        married = float(form.get("Married"))
        education = float(form.get("Education"))
        applicant_income = float(form.get("ApplicantIncome"))
        coapplicant_income = float(form.get("CoapplicantIncome"))
        loan_amount = float(form.get("LoanAmount"))
        loan_amount_term = float(form.get("Loan_Amount_Term"))
        credit_history = float(form.get("Credit_History"))

        if not applicant_name:
            raise ValueError("Name is required")

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

        if approved:
            message = "Congratulations! Your loan application has been approved."
            description = (
                "Based on the details provided, your profile meets the "
                "criteria for loan approval. Our team will contact you "
                "shortly with the next steps."
            )
        else:
            message = "We're sorry, your loan application was not approved."
            description = (
                "Based on the details provided, your profile does not "
                "currently meet the criteria for approval. This may be "
                "due to factors like credit history or income relative "
                "to the requested loan amount. You're welcome to review "
                "your details and try again."
            )

        result = {
            "approved": approved,
            "name": applicant_name,
            "message": message,
            "description": description
        }

    except (TypeError, ValueError):
        result = {
            "approved": None,
            "name": applicant_name or "Applicant",
            "message": "Please check your input",
            "description": "Make sure your name is entered and all fields contain valid values, then try again."
        }

    return render_template("index.html", result=result, form_data=form)


if __name__ == "__main__":
    app.run(debug=True)