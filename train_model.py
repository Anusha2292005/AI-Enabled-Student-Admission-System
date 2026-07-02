import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = {
    'percentage': [85, 70, 90, 60, 95, 50, 80, 65],
    'entrance_score': [90, 65, 95, 55, 98, 45, 85, 60],
    'admission': [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['percentage', 'entrance_score']]
y = df['admission']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'admission_model.pkl')

print("Model trained successfully!")
