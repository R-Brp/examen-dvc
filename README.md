# Examen MLOps — DVC & DagsHub

**Auteure :** Romane Beaurepere  
**Email :** beaurepere.romane@gmail.com  
**GitHub :** https://github.com/R-Brp/examen-dvc  
**DagsHub :** https://dagshub.com/R-Brp/examen-dvc

---

## Objectif

Mettre en place un pipeline de Machine Learning reproductible avec DVC sur un jeu de données de régression, en versionnant les données et les modèles via DagsHub.

---

## Structure du projet

```
examen-dvc/
├── data/
│   └── processed/          # Données générées par le pipeline (versionnées avec DVC)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── X_train_scaled.csv
│       └── X_test_scaled.csv
├── models/                 # Artefacts du modèle (versionnés avec DVC)
│   ├── scaler.pkl
│   ├── best_params.pkl
│   └── trained_model.pkl
├── metrics/
│   └── scores.json         # Métriques finales (trackées par DVC)
├── src/
│   ├── data/
│   │   ├── split_data.py
│   │   └── normalize_data.py
│   └── models/
│       ├── grid_search.py
│       ├── train_model_final.py
│       └── evaluate_model_final.py
├── dvc.yaml                # Définition du pipeline DVC
├── dvc.lock                # Snapshot reproductible du pipeline
└── .dvc/config             # Configuration du remote DagsHub
```

---

## Pipeline DVC

Le pipeline est composé de 5 étapes chaînées définies dans `dvc.yaml` :

### 1. `split` — Chargement et découpage des données
**Script :** `src/data/split_data.py`

Télécharge le dataset brut depuis S3, puis effectue un split 80/20 (train/test) avec `random_state=42`.

**Sorties :** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

### 2. `normalize` — Normalisation des features
**Script :** `src/data/normalize_data.py`

Applique un `StandardScaler` (fit sur le train uniquement, transform sur train et test). Supprime la colonne `date` avant la normalisation. Sauvegarde le scaler pour réutilisation.

**Sorties :** `X_train_scaled.csv`, `X_test_scaled.csv`, `models/scaler.pkl`

### 3. `grid_search` — Recherche des hyperparamètres
**Script :** `src/models/grid_search.py`

Lance un `GridSearchCV` (cross-validation à 3 folds) sur un `RandomForestRegressor` avec la grille :
- `n_estimators` : [50, 100, 200]
- `max_depth` : [10, 20, None]
- `min_samples_split` : [2, 5]

**Sortie :** `models/best_params.pkl`

### 4. `train` — Entraînement du modèle final
**Script :** `src/models/train_model_final.py`

Entraîne un `RandomForestRegressor` avec les meilleurs hyperparamètres trouvés à l'étape précédente.

**Sortie :** `models/trained_model.pkl`

### 5. `evaluate` — Évaluation du modèle
**Script :** `src/models/evaluate_model_final.py`

Calcule les métriques sur le jeu de test et les sauvegarde dans `metrics/scores.json`.

**Sortie :** `metrics/scores.json`

---

## Métriques obtenues

| Métrique | Valeur |
|----------|--------|
| MSE      | 0.809  |
| RMSE     | 0.899  |
| R²       | 0.192  |
| MAE      | 0.694  |

---

## Reproduire le pipeline

```bash
# Cloner le repo
git clone https://github.com/R-Brp/examen-dvc.git
cd examen-dvc

# Installer les dépendances
pip install dvc scikit-learn pandas numpy

# Lancer le pipeline complet
dvc repro

# Voir les métriques
dvc metrics show
```

---

## Remote storage — DagsHub

Les données et modèles sont versionnés sur DagsHub via le remote DVC configuré dans `.dvc/config` :

```
https://dagshub.com/R-Brp/examen-dvc.dvc
```

Pour pousser les artefacts :

```bash
dvc remote modify --local dagshub user R-Brp
dvc remote modify --local dagshub password <VOTRE_TOKEN>
dvc push
```

---

## Source des données

Dataset brut : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv
