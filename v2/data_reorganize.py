import argparse
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd


def extract_patient_number(filename: str) -> Optional[int]:
	"""Extract 4-digit patient number surrounded by underscores from a filename.

	Example matches: _0001_, _1234_
	Returns None if not found.
	"""
	match = re.search(r"_(\d{4})_", filename)
	return int(match.group(1)) if match else None


def list_csv_files(directory_path: str) -> List[str]:
	return [
		os.path.join(directory_path, file_name)
		for file_name in os.listdir(directory_path)
		if file_name.lower().endswith(".csv")
	]


def determine_label(
	file_name: str,
	patient_number: Optional[int],
	disease_df: pd.DataFrame,
	exclude_healthy: bool,
	healthy_label: int,
) -> Optional[int]:
	"""Return the label for this file or None if the file should be skipped."""
	name_lower = os.path.basename(file_name).lower()
	if "healthy" in name_lower:
		if exclude_healthy:
			return None
		return healthy_label

	if "patho" in name_lower and patient_number is not None:
		matched = disease_df[disease_df["Patient"] == patient_number]
		if not matched.empty:
			return matched.iloc[0]["Score (type)"]

	# Otherwise unknown
	return -1


def process_all_files(
	dir_path: str,
	disease_excel_path: str,
	output_csv: str,
	exclude_healthy: bool = False,
	healthy_label: int = 6,
) -> None:
	"""Aggregate columns from frequency CSVs into a single dataset with labels.

	Each input CSV is expected to have frequency measurements by column. Columns
	containing any NaN values are skipped. The label is appended as the last
	value per output row.
	"""

	if not os.path.exists(dir_path):
		raise FileNotFoundError(f"Directory not found: {dir_path}")
	if not os.path.exists(disease_excel_path):
		raise FileNotFoundError(f"Excel file not found: {disease_excel_path}")

	# Read disease/score mapping
	disease_df = pd.read_excel(disease_excel_path)
	if "Patient" not in disease_df.columns or "Score (type)" not in disease_df.columns:
		raise ValueError(
			"Excel must contain 'Patient' and 'Score (type)' columns."
		)

	all_rows: List[np.ndarray] = []
	for csv_path in list_csv_files(dir_path):
		patient_number = extract_patient_number(os.path.basename(csv_path))
		label = determine_label(
			file_name=csv_path,
			patient_number=patient_number,
			disease_df=disease_df,
			exclude_healthy=exclude_healthy,
			healthy_label=healthy_label,
		)

		# Skip if excluded (e.g., healthy when exclude_healthy=True)
		if label is None:
			continue

		# Read frequency DataFrame; columns are samples to transform into rows
		freq_df = pd.read_csv(csv_path, index_col=0)

		for column_name in freq_df.columns:
			column_values = freq_df[column_name]
			if column_values.isna().any():
				continue
			row_with_label = np.append(column_values.values, label)
			all_rows.append(row_with_label)

	if not all_rows:
		raise ValueError("No valid data rows were produced. Check inputs and file patterns.")

	combined_df = pd.DataFrame(all_rows)
	# Write without header/index to mimic notebook behavior
	combined_df.to_csv(output_csv, index=False, header=False)
	print(f"Processed data saved to {output_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Aggregate frequency CSV columns into a single dataset with labels, "
			"optionally excluding healthy samples."
		)
	)
	parser.add_argument(
		"--dir",
		dest="dir_path",
		default=os.path.join("Skin", "Resultados"),
		help="Directory containing input CSV files (default: Skin/Resultados)",
	)
	parser.add_argument(
		"--excel",
		dest="disease_excel_path",
		default=os.path.join("Skin", "0list.xlsx"),
		help="Excel file path with columns 'Patient' and 'Score (type)' (default: Skin/0list.xlsx)",
	)
	parser.add_argument(
		"--output",
		dest="output_csv",
		default=os.path.join("v2", "workingdataset3.csv"),
		help="Output CSV path (default: v2/workingdataset3.csv)",
	)
	parser.add_argument(
		"--exclude-healthy",
		action="store_true",
		help="Exclude samples whose filename contains 'healthy'",
	)
	parser.add_argument(
		"--healthy-label",
		type=int,
		default=6,
		help="Label to assign to healthy samples when included (default: 6)",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	process_all_files(
		dir_path=args.dir_path,
		disease_excel_path=args.disease_excel_path,
		output_csv=args.output_csv,
		exclude_healthy=args.exclude_healthy,
		healthy_label=args.healthy_label,
	)


if __name__ == "__main__":
	main()


