# SCRIPT 1: Split des donnees en train/test

import pandas as pd
from sklearn.model_selection import train_test_split
import os

url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
print("Telechargement des donnees...")
df = pd.read_csv(url)

print("Dataset charge: " + str(df.shape[0]) + " lignes, " + str(df.shape[1]) + " colonnes")
print(df.head())

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("X (features): " + str(X.shape))
print("y (target): " + str(y.shape))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("X_train: " + str(X_train.shape))
print("X_test: " + str(X_test.shape))
print("y_train: " + str(y_train.shape))
print("y_test: " + str(y_test.shape))

output_dir = "data/processed"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Sauvegarde des donnees...")
X_train.to_csv(output_dir + "/X_train.csv", index=False)
X_test.to_csv(output_dir + "/X_test.csv", index=False)
y_train.to_csv(output_dir + "/y_train.csv", index=False)
y_test.to_csv(output_dir + "/y_test.csv", index=False)

print("Fichiers sauvegardes dans " + output_dir + "/")
print("Script termine!")
