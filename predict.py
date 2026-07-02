import joblib

model = joblib.load('admission_model.pkl')

percentage = 88
entrance_score = 92

prediction = model.predict([[percentage, entrance_score]])

if prediction[0] == 1:
    print("Eligible for Admission")
else:
    print("Not Eligible")
