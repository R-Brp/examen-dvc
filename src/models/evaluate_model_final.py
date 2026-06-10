import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
import numpy as np
import os

print("Chargement des donnees...")
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

print("Chargement du modele...")
with open("models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Predictions...")
y_pred = model.predict(X_test)

print("Calcul des metriques...")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MSE: " + str(mse))
print("RMSE: " + str(rmse))
print("R2: " + str(r2))
print("MAE: " + str(mae))

os.makedirs("metrics", exist_ok=True)

metrics = {
    "mse": float(mse),
    "rmse": float(rmse),
    "r2": float(r2),
    "mae": float(mae)
}

with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f)

print("Metriques sauvegardees!")
