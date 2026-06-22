# Loan Approval Prediction — Machine Learning Coursework

A complete machine learning pipeline that predicts whether a loan application will be approved or rejected, based on applicant financial and demographic data.

---

## Problem Type

Binary classification — predicts `Loan_Status` as **Approved** or **Rejected**.

---

## Project Structure

```text
ml-coursework/

├── dataset/
│   └── loan_approval_dataset.csv      # Raw dataset
├── notebooks/
│   └── loan_approval_prediction.ipynb # Full ML pipeline (EDA → deployment)
├── deployment/
│   ├── model.pkl                      # Trained Random Forest model
│   ├── scaler.pkl                     # Fitted StandardScaler
│   ├── predictor.py                   # Command-line predictor script
│   ├── app.py                         # Flask web app
│   ├── templates/
│   │   └── index.html                 # Web form
│   └── static/
│       └── style.css                  # Styling
├── report/                            # Written report (PDF)
├── screenshots/                       # Evidence screenshots for report
└── README.md
```

---

## Pipeline Summary

### 1. Data Collection

Loan approval dataset (CSV) with 12 original features and a binary target (`Loan_Status`).

### 2. Data Exploration & Preprocessing

Missing value imputation (median/mode), categorical encoding, distribution analysis via EDA plots.

### 3. Feature Engineering

Created `TotalIncome` (Applicant + Coapplicant Income). Applied `SelectKBest` (chi-square) to reduce from 12 to the 8 strongest features.

### 4. Model Training

Three models trained and compared:

* Logistic Regression (basic model)
* K-Nearest Neighbors (distance-based model)
* Random Forest (ensemble model)

### 5. Hyperparameter Tuning

`GridSearchCV` applied to both KNN (`n_neighbors`) and Random Forest (`n_estimators`, `max_depth`, `min_samples_split`), since Random Forest is the model ultimately deployed.

### 6. Model Evaluation

Accuracy, Precision, Recall, and F1-score computed for all three models.

### 7. Model Comparison

Best model selected based on the full metrics comparison table (see notebook output / report).

### 8. Deployment

Random Forest model deployed two ways:

* `predictor.py`: command-line script
* `app.py`: Flask web GUI

---

## Final Features Used by the Model

After feature selection, the model is trained on 8 features (in this exact order):

```text
CoapplicantIncome
TotalIncome
ApplicantIncome
LoanAmount
Credit_History
Loan_Amount_Term
Education
Married
```

`Gender`, `Dependents`, `Self_Employed`, and `Property_Area` were dropped after scoring lower on the chi-square feature selection test.

---

## How to Run

### 1. Run the Notebook (Trains and Saves the Model)

```bash
cd notebooks
jupyter notebook loan_approval_prediction.ipynb
```

Run all cells top to bottom. This saves the trained `model.pkl` and `scaler.pkl` into the `deployment/` folder.

---

### 2. Run the Command-Line Predictor

```bash
cd deployment
pip install joblib pandas scikit-learn
python predictor.py
```

You'll be prompted for 7 inputs (`TotalIncome` is calculated automatically):

* Coapplicant Income
* Applicant Income
* Loan Amount
* Credit History (1 = Good, 0 = Bad)
* Loan Amount Term
* Education (1 = Graduate, 0 = Not Graduate)
* Married (1 = Yes, 0 = No)

---

### 3. Run the Flask Web App

```bash
cd deployment
pip install flask joblib pandas scikit-learn
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

in your browser.

Fill in the form and click **Predict** to see:

* Loan Approved
* Loan Rejected

---

## Notes

* Make sure you run the notebook **before** running `predictor.py` or `app.py`, since both load `model.pkl` and `scaler.pkl` from the `deployment/` folder.

* If you see a scikit-learn `InconsistentVersionWarning` when loading the model, it's safe to ignore as long as predictions still run — it just means the installed scikit-learn version differs slightly from the one used to train the model.

* Column order in `predictor.py` and `app.py` must exactly match the order used during training (see **Final Features Used by the Model** above), or `scaler.transform()` will raise a shape-mismatch error.
