import joblib
import pandas as pd


model = joblib.load("./model/loan_default.pkl")
 
# Create test sample data diagnostics on startup
test_df = joblib.load("./data/x_test_sample.csv")
x_test_sample = test_df.sample(1)
print("test sample:", x_test_sample) 

# Ensure diagnostic sample matches training feature expectations
expected_features = [
    "Current Loan Amount", "Term", "Credit Score", "Annual Income",
    "Years in current job", "Home Ownership", "Purpose", "Monthly Debt",
    "Years of Credit History", "Months since last delinquent",
    "Number of Open Accounts", "Number of Credit Problems",
    "Current Credit Balance", "Maximum Open Credit", "Bankruptcies", "Tax Liens"
]
x_test_sample = x_test_sample[expected_features]

yhat = model.predict(x_test_sample)
print("Available columns in your data:\n", list(x_test_sample.columns))
print(f"Prediction: {yhat}")

