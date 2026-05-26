from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('Model/loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            int(request.form['dependents']),
            int(request.form['education']),
            int(request.form['self_employed']),
            int(request.form['income']),
            int(request.form['loan_amount']),
            int(request.form['loan_term']),
            int(request.form['cibil']),
            int(request.form['residential']),
            int(request.form['commercial']),
            int(request.form['luxury']),
            int(request.form['bank'])
        ]

        prediction = model.predict([features])

        result = "Loan Approved ✅" if prediction[0] == 1 else "Loan Rejected ❌"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)