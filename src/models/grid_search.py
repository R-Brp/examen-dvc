import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import os

print("Chargement des donnees...")
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").iloc[:, 0]

print("X_train: " + str(X_train.shape))
print("y_train: " + str(y_train.shape))

print("GridSearch en cours...")
model = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

print("Meilleurs parametres: " + str(grid.best_params_))
print("Meilleur score: " + str(grid.best_score_))

os.makedirs("models", exist_ok=True)
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid.best_params_, f)

print("Best params sauvegardes!")
