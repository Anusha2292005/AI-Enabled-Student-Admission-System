from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load("admission_model.pkl")
except:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    percentage = float(request.form['percentage'])
    entrance_score = float(request.form['entrance_score'])

    if model:
        prediction = model.predict([[percentage, entrance_score]])
        result = "Eligible for Admission" if prediction[0] == 1 else "Not Eligible"
    else:
        result = "Model not trained yet."

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
