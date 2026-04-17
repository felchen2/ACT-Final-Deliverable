import pandas as pd
import numpy as np

# =========================
# INPUTS
# =========================
FILES = [
    "sa_exit_master_long.csv",                    # from 2008-2013(1).py
    "NU_FINAL/NU_exit_master_long_2016format.csv",# from 2014-2019(1).py
    "sa_exit_summary_2021_2023.csv",              # from 2021-2023 copy.py
    "sa_exit_master_long_2024_2025.csv",          # from 2024-25 copy.py
]

# Use this as the schema/template (column order)
TEMPLATE = "sa_exit_master_long.csv"

# =========================
# OUTPUT
# =========================
OUT_PATH = "sa_exit_MASTER_merged_all4.csv"

# =========================
# LOAD TEMPLATE COLUMNS
# =========================
template_df = pd.read_csv(TEMPLATE)
MASTER_COLS = list(template_df.columns)

def load_and_align(path: str, master_cols):
    df = pd.read_csv(path)

    # add any missing master columns
    for c in master_cols:
        if c not in df.columns:
            df[c] = np.nan

    # keep strict schema + order
    df = df[master_cols].copy()

    # light cleanup
    if "ResponseText" in df.columns:
        df["ResponseText"] = df["ResponseText"].astype(str).str.strip()
        df.loc[df["ResponseText"].str.lower().isin(["nan", "none", ""]), "ResponseText"] = np.nan

    return df

# =========================
# LOAD + CONCAT ALL
# =========================
frames = []
for f in FILES:
    aligned = load_and_align(f, MASTER_COLS)
    aligned["__source"] = f  # optional trace column
    frames.append(aligned)

merged = pd.concat(frames, ignore_index=True)

# =========================
# DROP DUPLICATES
# =========================
merged_no_source = merged.drop(columns=["__source"]).drop_duplicates()

# =========================
# SAVE
# =========================
merged_no_source.to_csv(OUT_PATH, index=False)

print("Wrote:", OUT_PATH)
print("Row counts by file:")
for f, df in zip(FILES, frames):
    print(f" - {f}: {len(df):,}")
print("Merged rows (after dedupe):", f"{len(merged_no_source):,}")
