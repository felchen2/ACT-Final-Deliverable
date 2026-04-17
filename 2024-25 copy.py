# =========================
# 2024-2025.py  (FULL SCRIPT)
# =========================
import re
from pathlib import Path
import pandas as pd


def infer_year_from_filename(fp: Path) -> int | None:
    m = re.search(r"(19|20)\d{2}", fp.name)
    return int(m.group()) if m else None


def detect_header_row(df: pd.DataFrame) -> int:
    """
    Finds the first row that looks like a true respondent-level header row.
    """
    best_idx = None
    best_score = -1

    for i in range(min(len(df), 40)):
        vals = [x for x in df.iloc[i].tolist() if pd.notna(x)]
        vals = [str(x).strip() for x in vals if str(x).strip()]

        if not vals:
            continue

        score = 0
        for v in vals:
            vl = v.lower()
            if len(v) > 15:
                score += 1
            if "submitted answers" in vl:
                score -= 10
            if "questions:" in vl:
                score -= 10
            if v.lower() in {"label", "question", "responses"}:
                score -= 10

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        raise ValueError("Could not detect header row.")
    return best_idx


def parse_raw_sheet(file_path: str) -> pd.DataFrame:
    fp = Path(file_path)
    year = infer_year_from_filename(fp)

    raw = pd.read_excel(fp, sheet_name=0, header=None)

    header_row = detect_header_row(raw)
    headers = raw.iloc[header_row].tolist()

    data = raw.iloc[header_row + 1:].copy()
    data.columns = headers
    data = data.dropna(how="all").reset_index(drop=True)

    data["Year"] = year
    data["SourceFile"] = fp.name
    data["ResponseID"] = [f"{fp.stem}-{i+1}" for i in range(len(data))]

    id_cols = ["Year", "SourceFile", "ResponseID"]
    value_cols = [c for c in data.columns if c not in id_cols]

    long = data.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="QuestionText",
        value_name="ResponseText"
    )

    long = long.dropna(subset=["ResponseText"]).copy()
    long["ResponseText"] = long["ResponseText"].astype(str).str.strip()
    long = long[long["ResponseText"] != ""]

    long["QuestionRaw"] = long["QuestionText"]
    long["QuestionNumber"] = pd.NA
    long["Section"] = pd.NA
    long["ResponseType"] = "Unknown"

    keep_cols = [
        "Year", "SourceFile", "ResponseID",
        "QuestionRaw", "QuestionNumber", "QuestionText", "Section",
        "ResponseType", "ResponseText"
    ]
    return long[keep_cols]


def combine_years(file_paths: list[str]) -> pd.DataFrame:
    frames = []
    for fp in file_paths:
        frames.append(parse_raw_sheet(fp))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    files = [
        # r"2024-2025/Study Abroad Questionnaire (Fall 2024).xlsx",
        # r"2024-2025/Study Abroad Questionnaire (Spring 1 2025).xlsx",
        # r"2024-2025/Study Abroad Questionnaire (Summer 2025).xlsx",
        r"",
    ]

    master = combine_years(files)
    master.to_csv("sa_exit_master_long_2024_2025.csv", index=False)
    master.to_parquet("sa_exit_master_long_2024_2025.parquet", index=False)

    print("Wrote:", "sa_exit_master_long_2024_2025.csv", "and", "sa_exit_master_long_2024_2025.parquet")