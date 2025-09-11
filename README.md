# Elastography + AI

- \nalysis.py\ : visualisation, extraction de features.
- \predict.py\ : modèles (ridge, lasso, RF, PLS) + LOOCV.

## Lancer
python analysis.py --select
python predict.py --model ridge --annotate --clip-min 0 --clip-max 5
