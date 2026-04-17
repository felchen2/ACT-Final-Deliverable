import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_likert(response_text):
    if pd.isna(response_text):
        return np.nan
    match = re.match(r"\((\d+)\)", str(response_text).strip())
    if match:
        return float(match.group(1))
    return np.nan


def categorize_safety_question(question_text):
    q = str(question_text)

    if re.search(r"safe on ACT campus", q, re.I):
        return "ACT Campus"
    if re.search(r"safe in (my )?housing|Residence Hall|hotel", q, re.I):
        return "Housing"
    if re.search(r"safe in Thessaloniki", q, re.I):
        return "Thessaloniki"
    if re.search(r"traveling through Greece", q, re.I):
        return "Travel in Greece"

    return None


def build_safety_dataset(input_csv, output_dir):
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    df["Year_num"] = pd.to_numeric(df["Year"], errors="coerce")

    safety = df[df["QuestionText"].astype(str).str.contains("safe", case=False, na=False)].copy()
    safety["category"] = safety["QuestionText"].apply(categorize_safety_question)
    safety["code"] = safety["ResponseText"].apply(parse_likert)

    safety = safety[
        safety["category"].notna() &
        safety["code"].isin([1, 2, 3, 4, 5])
    ].copy()

    safety["positive"] = safety["code"].isin([1, 2]).astype(int)
    safety["strongly_agree"] = safety["code"].eq(1).astype(int)
    safety["negative"] = safety["code"].isin([4, 5]).astype(int)

    cleaned_path = output_dir / "cleaned_safety_dataset.csv"
    safety.to_csv(cleaned_path, index=False)

    summary = (
        safety.groupby(["Year_num", "category"])
        .agg(
            n=("code", "size"),
            positive_pct=("positive", "mean"),
            strongly_agree_pct=("strongly_agree", "mean"),
            negative_pct=("negative", "mean"),
            mean_code=("code", "mean"),
        )
        .reset_index()
        .sort_values(["Year_num", "category"])
    )

    summary_path = output_dir / "safety_summary_by_year.csv"
    summary.to_csv(summary_path, index=False)

    return cleaned_path, summary_path
