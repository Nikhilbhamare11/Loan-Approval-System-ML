import pickle

# Load model
model = pickle.load(open("Model/loan_model.pkl", "rb"))

# Input
loanapp = [[
    2,   # no_of_dependents
    1,   # education (1 = Graduate)
    0,   # self_employed (0 = No)
    5000000,   # income_annum
    20000000,  # loan_amount
    10,        # loan_term
    750,       # cibil_score
    3000000,   # residential_assets_value
    2000000,   # commercial_assets_value
    5000000,   # luxury_assets_value
    4000000    # bank_asset_value
   ]]

loanrej = [[5,0,1,2000000,30000000,20,350,1000000,500000,2000000,1000000]]

# Prediction
result = model.predict(loanapp)

# Output
if result[0] == 1:
    print("Loan Approved!!")
else:
    print("Loan Rejected....")

