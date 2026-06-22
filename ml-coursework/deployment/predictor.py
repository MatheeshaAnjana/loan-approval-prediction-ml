import joblib
import numpy as np

model = joblib.load(
    "model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

print(
    "\nLoan Approval Predictor\n"
)

Gender = float(
    input(
        "Gender (1 Male,0 Female): "
    )
)

Married = float(
    input(
        "Married (1 Yes,0 No): "
    )
)

ApplicantIncome = float(
    input(
        "Applicant Income: "
    )
)

LoanAmount = float(
    input(
        "Loan Amount: "
    )
)

CreditHistory = float(
    input(
        "Credit History: "
    )
)

PropertyArea = float(
    input(
        "Property Area: "
    )
)

sample = np.array([[

    Gender,

    Married,

    ApplicantIncome,

    LoanAmount,

    CreditHistory,

    PropertyArea

]])

sample = scaler.transform(
    sample
)

prediction = model.predict(
    sample
)

if prediction[0] == 1:

    print(
        "\nLoan Approved"
    )

else:

    print(
        "\nLoan Rejected"
    )