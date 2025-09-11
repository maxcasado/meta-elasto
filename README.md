# Elastography + AI

Visualisation des mesures Healthy/Patho par patient, extraction de features frugales et prédiction du **score (0–5)** avec plusieurs modèles simples adaptés au **small data**.

---

## Arborescence attendue

```
.
├── analysis.py        # Visualisation + extraction de features + PCA
├── predict.py         # Modèles ML interchangeables + LOOCV
├── Skin/
│   ├── 0list.xlsx     # Col. A = patient id (entier), Col. B = score (0–5)
│   └── Resultados/
│       ├── Elastome_0001_Healthy_angle_1.csv
│       ├── Elastome_0001_Patho_angle_1.csv
│       ├── Elastome_0002_Healthy_angle_1.csv
│       └── Elastome_0002_Patho_angle_1.csv
└── README.md
```

**Format des CSV** : chaque ligne = un instant t ; colonnes = fréquences (certaines peuvent contenir `NaN`).

---

## Installation rapide

> Python 3.10+ recommandé (ok avec 3.13). Besoin d’`openpyxl` pour lire l’Excel.

```bash
# (optionnel) créer un venv
python -m venv .venv
# activer le venv
# Windows PowerShell :
. .\.venv\Scripts\Activate.ps1
# macOS/Linux :
# source .venv/bin/activate

# installer les dépendances
pip install numpy pandas matplotlib scikit-learn openpyxl
```

### (Optionnel) `requirements.txt`
Crée un fichier `requirements.txt` avec :
```
numpy
pandas
matplotlib
scikit-learn
openpyxl
```
Puis :
```bash
pip install -r requirements.txt
```

---

## Ignorer les données (`Skin/`) dans Git

Ajoute au `.gitignore` :
```
# Données locales
Skin/
```

Si `Skin/` a déjà été indexé :
```bash
git rm -r --cached Skin
git commit -m "chore: ignore Skin data directory"
git push
```

---

## `analysis.py` — Visualisation & PCA

Afficher un patient précis (ID entier **ou** au format `0007`) :
```bash
python analysis.py --patient 7
# ou
python analysis.py --patient 0007
```

Tracer le **premier** patient trouvé si aucun argument :
```bash
python analysis.py
```

Changer le nombre de courbes de fréquences superposées :
```bash
python analysis.py --patient 7 --k_lines 8
```

Vue **globale** (features agrégés) avec **PCA** :
```bash
python analysis.py --pca
```

Ce que fait `analysis.py` :
- lit `Skin/0list.xlsx`,
- charge `Healthy` & `Patho` si dispo,
- convertit toutes les colonnes en numérique, masque `±inf`,
- sélectionne automatiquement les fréquences **les moins lacunaires**,
- trace séries temporelles Healthy vs Patho, **heatmaps**, et **delta des moyennes**,
- fabrique des **features frugales** (moyenne/écart-type + deltas) réutilisées par `predict.py`.

---

## `predict.py` — Modélisation & évaluation LOOCV

Le script construit le dataset via `analysis.py`, fait une **Leave-One-Out CV**, affiche :
- un tableau `patient | y_true | y_pred`,
- R² et RMSE,
- un nuage **réel vs prédit** (diagonale idéale) avec **annotations** optionnelles.

### Utilisation de base (Ridge + annotations + bornage 0–5)
```bash
python predict.py --annotate --clip-min 0 --clip-max 5
```

### Choisir le modèle
```bash
# Ridge (par défaut)
python predict.py --model ridge

# Lasso
python predict.py --model lasso

# ElasticNet
python predict.py --model elastic

# PLS (Partial Least Squares)
python predict.py --model pls --pls-components 2

# Random Forest
python predict.py --model rf --rf-estimators 300 --rf-min-samples-leaf 2
```

### Options utiles
- `--k-best K` : garde les **K** meilleures features (sélection univariée, f_regression).
- `--pca-components P` : applique une **PCA** (réduction de dimension) avant le modèle.
- `--ignore 7 12` : **exclut** certains patients (données foireuses/outliers).
- `--clip-min 0 --clip-max 5` : borne les prédictions à l’intervalle [0,5].
- `--annotate` : affiche l’**ID** du patient à côté de chaque point.

### Exemples
```bash
# Random Forest + annotations + ignorer patient 7 + bornage 0–5
python predict.py --model rf --rf-estimators 300 --rf-min-samples-leaf 2 --annotate --ignore 7 --clip-min 0 --clip-max 5

# Lasso avec sélection des 8 meilleures features
python predict.py --model lasso --k-best 8 --annotate --clip-min 0 --clip-max 5

# Ridge avec PCA(5) + annotations
python predict.py --model ridge --pca-components 5 --annotate
```

---

## Bonnes pratiques (small data, ~20–25 patients)

- **Toujours normaliser** pour les modèles linéaires (fait automatiquement dans le pipeline).
- **Réduire la dimension** (PCA ou `--k-best`) → viser **3–8 features effectives**.
- **Régulariser** : Ridge/Lasso/ElasticNet plutôt que OLS.
- **PLS** marche bien quand `n << p`.
- **Comparer à la baseline** (prédire la moyenne) : le script affiche le **Baseline RMSE**.
- **Inspecter les NaN** : trop de colonnes quasi vides => bruit → préférer les deltas de moyenne sur les fréquences bien couvertes.
- **Outliers** : utiliser `--ignore` pour exclure temporairement un patient suspect et valider la stabilité.

---


