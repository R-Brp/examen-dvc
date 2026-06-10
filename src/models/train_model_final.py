import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

print("Chargement des donnees...")
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").iloc[:, 0]

print("Chargement des meilleurs parametres...")
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

print("Parametres: " + str(best_params))

print("Entraînement du modele...")
model = RandomForestRegressor(random_state=42, **best_params)
model.fit(X_train, y_train)

print("Score entraînement: " + str(model.score(X_train, y_train)))

with open("models/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modele sauvegarde!")
