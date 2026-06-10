import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("Chargement des donnees...")
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

print("Colonnes: " + str(X_train.columns.tolist()))

X_train = X_train.drop("date", axis=1)
X_test = X_test.drop("date", axis=1)

print("X_train sans date: " + str(X_train.shape))
print("X_test sans date: " + str(X_test.shape))

scaler = StandardScaler()

print("Normalisation...")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("data/processed/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("data/processed/X_test_scaled.csv", index=False)

os.makedirs("models", exist_ok=True)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Scaler sauvegarde dans models/scaler.pkl")
print("Fait!")
