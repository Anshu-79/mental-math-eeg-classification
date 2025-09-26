import os
import pandas as pd

# Paths
records_file = "data/RECORDS"  # Contains EDF filenames
subject_info_csv = "data/subject-info.csv"  # Contains subject info
output_csv = "data/subject_csv.csv"  # CSV to generate

# Read EDF filenames
with open(records_file, "r") as f:
    edf_files = [line.strip() for line in f.readlines() if line.strip()]

# Read subject info
subj_info = pd.read_csv(subject_info_csv)

# Map subject to condition (example: 0 if Count quality == 0, else 1)
subj_info["condition"] = subj_info["Count quality"].apply(lambda x: 0 if x == 0 else 1)

rows = []
for edf_file in edf_files:
    # EDF filename like Subject00_1.edf
    subj_name = "_".join(edf_file.split("_")[:1])  # Subject00
    condition_row = subj_info[subj_info["Subject"] == subj_name]
    if len(condition_row) == 0:
        raise ValueError(f"Subject {subj_name} not found in subject-info.csv")
    condition = int(condition_row["condition"].values[0])
    rows.append({"file_name": edf_file, "subject": subj_name, "condition": condition})

# Save CSV
df_out = pd.DataFrame(rows)
df_out.to_csv(output_csv, index=False)
print(f"Generated subject CSV: {output_csv}")
