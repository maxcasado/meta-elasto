import argparse
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	roc_auc_score,
)
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split, LeaveOneOut
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib.pyplot as plt
from itertools import cycle


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"Dataset not found: {csv_path}")
	data = pd.read_csv(csv_path, header=None)
	if data.shape[1] < 2:
		raise ValueError("Dataset must have at least 2 columns (features + label)")
	X = data.iloc[:, :-1].values
	y = data.iloc[:, -1].values
	return X, y


def ensure_label_integers(y: np.ndarray) -> Tuple[np.ndarray, LabelEncoder]:
	if np.issubdtype(y.dtype, np.number):
		return y.astype(int), None
	encoder = LabelEncoder()
	return encoder.fit_transform(y), encoder


def train_lightgbm(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_val: np.ndarray,
	y_val: np.ndarray,
	params: Dict,
):
	try:
		from lightgbm import LGBMClassifier
	except ImportError as exc:
		raise ImportError("lightgbm is not installed. Add it to requirements and install.") from exc
	model = LGBMClassifier(**params)
	model.fit(
		X_train,
		y_train,
		eval_set=[(X_val, y_val)],
	)
	return model


def train_xgboost(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_val: np.ndarray,
	y_val: np.ndarray,
	params: Dict,
):
	try:
		from xgboost import XGBClassifier
	except ImportError as exc:
		raise ImportError("xgboost is not installed. Add it to requirements and install.") from exc
	model = XGBClassifier(**params)
	model.fit(
		X_train,
		y_train,
		eval_set=[(X_val, y_val)],
	)
	return model


def evaluate_model(model, X_test, y_test, average: str = "macro") -> Dict:
	y_pred = model.predict(X_test)
	metrics = {
		"accuracy": float(accuracy_score(y_test, y_pred)),
		"f1_macro": float(f1_score(y_test, y_pred, average="macro")),
		"f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
		"confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
		"classification_report": classification_report(y_test, y_pred),
	}
	return metrics


def run_loocv(X, y, lgbm_params: Dict, xgb_params: Dict, num_classes: int) -> Dict:
	"""Run Leave-One-Out Cross-Validation and return aggregated metrics."""
	loo = LeaveOneOut()
	
	lgbm_accuracies = []
	lgbm_f1_macros = []
	lgbm_f1_weighteds = []
	
	xgb_accuracies = []
	xgb_f1_macros = []
	xgb_f1_weighteds = []
	
	all_y_true_lgbm = []
	all_y_pred_lgbm = []
	all_y_true_xgb = []
	all_y_pred_xgb = []
	
	fold_count = 0
	total_folds = len(X)
	
	for train_idx, test_idx in loo.split(X):
		fold_count += 1
		if fold_count % 50 == 0 or fold_count == total_folds:
			print(f"LOOCV Progress: {fold_count}/{total_folds}")
		
		X_train_fold, X_test_fold = X[train_idx], X[test_idx]
		y_train_fold, y_test_fold = y[train_idx], y[test_idx]
		
		# Train models for this fold
		try:
			from lightgbm import LGBMClassifier
			lgbm_model = LGBMClassifier(**lgbm_params)
			lgbm_model.fit(X_train_fold, y_train_fold)
			
			from xgboost import XGBClassifier
			xgb_model = XGBClassifier(**xgb_params)
			xgb_model.fit(X_train_fold, y_train_fold)
		except Exception as e:
			print(f"Error in fold {fold_count}: {e}")
			continue
		
		# Predict
		y_pred_lgbm = lgbm_model.predict(X_test_fold)
		y_pred_xgb = xgb_model.predict(X_test_fold)
		
		# Store predictions for global confusion matrix
		all_y_true_lgbm.extend(y_test_fold)
		all_y_pred_lgbm.extend(y_pred_lgbm)
		all_y_true_xgb.extend(y_test_fold)
		all_y_pred_xgb.extend(y_pred_xgb)
		
		# Calculate fold metrics
		lgbm_acc = accuracy_score(y_test_fold, y_pred_lgbm)
		lgbm_f1_macro = f1_score(y_test_fold, y_pred_lgbm, average="macro", zero_division=0)
		lgbm_f1_weighted = f1_score(y_test_fold, y_pred_lgbm, average="weighted", zero_division=0)
		
		xgb_acc = accuracy_score(y_test_fold, y_pred_xgb)
		xgb_f1_macro = f1_score(y_test_fold, y_pred_xgb, average="macro", zero_division=0)
		xgb_f1_weighted = f1_score(y_test_fold, y_pred_xgb, average="weighted", zero_division=0)
		
		lgbm_accuracies.append(lgbm_acc)
		lgbm_f1_macros.append(lgbm_f1_macro)
		lgbm_f1_weighteds.append(lgbm_f1_weighted)
		
		xgb_accuracies.append(xgb_acc)
		xgb_f1_macros.append(xgb_f1_macro)
		xgb_f1_weighteds.append(xgb_f1_weighted)
	
	# Aggregate results
	results = {
		"lgbm": {
			"accuracy_mean": float(np.mean(lgbm_accuracies)),
			"accuracy_std": float(np.std(lgbm_accuracies)),
			"f1_macro_mean": float(np.mean(lgbm_f1_macros)),
			"f1_macro_std": float(np.std(lgbm_f1_macros)),
			"f1_weighted_mean": float(np.mean(lgbm_f1_weighteds)),
			"f1_weighted_std": float(np.std(lgbm_f1_weighteds)),
			"confusion_matrix": confusion_matrix(all_y_true_lgbm, all_y_pred_lgbm).tolist(),
			"classification_report": classification_report(all_y_true_lgbm, all_y_pred_lgbm),
		},
		"xgb": {
			"accuracy_mean": float(np.mean(xgb_accuracies)),
			"accuracy_std": float(np.std(xgb_accuracies)),
			"f1_macro_mean": float(np.mean(xgb_f1_macros)),
			"f1_macro_std": float(np.std(xgb_f1_macros)),
			"f1_weighted_mean": float(np.mean(xgb_f1_weighteds)),
			"f1_weighted_std": float(np.std(xgb_f1_weighteds)),
			"confusion_matrix": confusion_matrix(all_y_true_xgb, all_y_pred_xgb).tolist(),
			"classification_report": classification_report(all_y_true_xgb, all_y_pred_xgb),
		}
	}
	
	return results


def plot_roc_per_class(model, X_test, y_test, num_classes: int, output_path: str) -> None:
	"""Plot one-vs-rest ROC for each class and save figure."""
	if not hasattr(model, "predict_proba"):
		raise ValueError("Model lacks predict_proba required for ROC curves.")
	proba = model.predict_proba(X_test)
	y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))

	fpr_list = []
	tpr_list = []
	auc_list = []
	for class_index in range(num_classes):
		fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, class_index], proba[:, class_index])
		auc_i = auc(fpr_i, tpr_i)
		fpr_list.append(fpr_i)
		tpr_list.append(tpr_i)
		auc_list.append(auc_i)

	plt.figure(figsize=(8, 6))
	colors = cycle([
		"#1f77b4",
		"#ff7f0e",
		"#2ca02c",
		"#d62728",
		"#9467bd",
		"#8c564b",
		"#e377c2",
		"#7f7f7f",
		"#bcbd22",
		"#17becf",
	])
	for i, color in zip(range(num_classes), colors):
		plt.plot(
			fpr_list[i],
			tpr_list[i],
			color=color,
			lw=2,
			label=f"Class {i} (AUC = {auc_list[i]:.3f})",
		)
	plt.plot([0, 1], [0, 1], "k--", lw=1)
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC per class (one-vs-rest)")
	plt.legend(loc="lower right", fontsize=8)
	os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def save_report(report_dir: str, name: str, metrics: Dict) -> None:
	os.makedirs(report_dir, exist_ok=True)
	# Save text report
	with open(os.path.join(report_dir, f"report_{name}.txt"), "w", encoding="utf-8") as f:
		f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
		f.write(f"F1 (macro): {metrics['f1_macro']:.4f}\n")
		f.write(f"F1 (weighted): {metrics['f1_weighted']:.4f}\n\n")
		f.write(metrics["classification_report"])  # includes per-class metrics

	# Save confusion matrix as CSV for convenience
	cm_path = os.path.join(report_dir, f"confusion_{name}.csv")
	cm_df = pd.DataFrame(metrics["confusion_matrix"])
	cm_df.to_csv(cm_path, index=False)


def save_loocv_report(report_dir: str, name: str, metrics: Dict) -> None:
	os.makedirs(report_dir, exist_ok=True)
	# Save text report with mean ± std
	with open(os.path.join(report_dir, f"report_{name}_loocv.txt"), "w", encoding="utf-8") as f:
		f.write(f"LEAVE-ONE-OUT CROSS-VALIDATION RESULTS\n")
		f.write(f"=====================================\n\n")
		f.write(f"Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}\n")
		f.write(f"F1 (macro): {metrics['f1_macro_mean']:.4f} ± {metrics['f1_macro_std']:.4f}\n")
		f.write(f"F1 (weighted): {metrics['f1_weighted_mean']:.4f} ± {metrics['f1_weighted_std']:.4f}\n\n")
		f.write("AGGREGATED CLASSIFICATION REPORT:\n")
		f.write(metrics["classification_report"])

	# Save confusion matrix as CSV
	cm_path = os.path.join(report_dir, f"confusion_{name}_loocv.csv")
	cm_df = pd.DataFrame(metrics["confusion_matrix"])
	cm_df.to_csv(cm_path, index=False)


def main():
	parser = argparse.ArgumentParser(
		description="Train and compare LightGBM and XGBoost on a labeled dataset"
	)
	parser.add_argument(
		"--data",
		dest="csv_path",
		default=os.path.join("v2", "workingdataset3.csv"),
		help="Path to dataset CSV (default: v2/workingdataset3.csv)",
	)
	parser.add_argument(
		"--test-size",
		type=float,
		default=0.2,
		help="Test split size (default: 0.2)",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Random seed (default: 42)",
	)
	parser.add_argument(
		"--reports-dir",
		dest="reports_dir",
		default=os.path.join("reports"),
		help="Directory to save reports (default: reports)",
	)
	parser.add_argument(
		"--save-models",
		action="store_true",
		help="Save trained models as .joblib files",
	)
	parser.add_argument(
		"--loocv",
		action="store_true",
		help="Use Leave-One-Out Cross-Validation instead of train/test split",
	)

	args = parser.parse_args()

	X, y = load_dataset(args.csv_path)
	y, label_encoder = ensure_label_integers(y)

	# infer number of classes
	num_classes = int(len(np.unique(y)))

	# Reasonable baseline params; can be tuned later
	lgbm_params = {
		"n_estimators": 500,
		"learning_rate": 0.05,
		"max_depth": -1,
		"num_leaves": 31,
		"subsample": 0.9,
		"colsample_bytree": 0.9,
		"objective": "multiclass",
		"num_class": num_classes,
		"random_state": args.random_state,
	}

	xgb_params = {
		"n_estimators": 500,
		"learning_rate": 0.05,
		"max_depth": 6,
		"subsample": 0.9,
		"colsample_bytree": 0.9,
		"objective": "multi:softprob",
		"eval_metric": "mlogloss",
		"random_state": args.random_state,
	}

	if args.loocv:
		print("Running Leave-One-Out Cross-Validation...")
		print(f"Total samples: {len(X)}")
		
		loocv_results = run_loocv(X, y, lgbm_params, xgb_params, num_classes)
		
		# Save LOOCV reports
		save_loocv_report(args.reports_dir, "lgbm_base", loocv_results["lgbm"])
		save_loocv_report(args.reports_dir, "xgb_base", loocv_results["xgb"])
		
		print("\nLOOCV Results:")
		print(
			"LightGBM — Accuracy: {:.4f}±{:.4f}, F1-macro: {:.4f}±{:.4f}".format(
				loocv_results["lgbm"]["accuracy_mean"], loocv_results["lgbm"]["accuracy_std"],
				loocv_results["lgbm"]["f1_macro_mean"], loocv_results["lgbm"]["f1_macro_std"]
			)
		)
		print(
			"XGBoost  — Accuracy: {:.4f}±{:.4f}, F1-macro: {:.4f}±{:.4f}".format(
				loocv_results["xgb"]["accuracy_mean"], loocv_results["xgb"]["accuracy_std"],
				loocv_results["xgb"]["f1_macro_mean"], loocv_results["xgb"]["f1_macro_std"]
			)
		)
		
	else:
		# Standard train/test split
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
		)

		# Train models
		lgbm_model = train_lightgbm(X_train, y_train, X_test, y_test, lgbm_params)
		xgb_model = train_xgboost(X_train, y_train, X_test, y_test, xgb_params)

		# Evaluate
		lgbm_metrics = evaluate_model(lgbm_model, X_test, y_test)
		xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

		# Save reports
		save_report(args.reports_dir, "lgbm_base", lgbm_metrics)
		save_report(args.reports_dir, "xgb_base", xgb_metrics)

		# ROC per-class plots
		roc_lgbm_path = os.path.join(args.reports_dir, "roc_lgbm_base.png")
		roc_xgb_path = os.path.join(args.reports_dir, "roc_xgb_base.png")
		plot_roc_per_class(lgbm_model, X_test, y_test, num_classes, roc_lgbm_path)
		plot_roc_per_class(xgb_model, X_test, y_test, num_classes, roc_xgb_path)

		if args.save_models:
			os.makedirs("models", exist_ok=True)
			joblib.dump(lgbm_model, os.path.join("models", "lgbm_model.joblib"))
			joblib.dump(xgb_model, os.path.join("models", "xgb_model.joblib"))

		print(
			"LightGBM — Accuracy: {:.4f}, F1-macro: {:.4f}".format(
				lgbm_metrics["accuracy"], lgbm_metrics["f1_macro"]
			)
		)
		print(
			"XGBoost  — Accuracy: {:.4f}, F1-macro: {:.4f}".format(
				xgb_metrics["accuracy"], xgb_metrics["f1_macro"]
			)
		)


if __name__ == "__main__":
	main()


