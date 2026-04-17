# =========================
# 2021-2023.py  (FULL SCRIPT)
# =========================
import re
from pathlib import Path
import pandas as pd


def infer_year_from_filename(fp: Path) -> int | None:
    m = re.search(r"(19|20)\d{2}", fp.name)
    return int(m.group()) if m else None


def parse_summary_sheet(file_path: str) -> pd.DataFrame:
    fp = Path(file_path)
    year = infer_year_from_filename(fp)

    raw = pd.read_excel(fp, sheet_name="Sheet1", header=None)

    records = []
    i = 0
    while i < len(raw):
        question_cell = raw.iloc[i, 1] if raw.shape[1] > 1 else None

        if pd.notna(question_cell):
            qtext = str(question_cell).strip()

            # skip header/meta rows
            if qtext not in {"Question", ""} and not qtext.startswith("Questions:"):
                # look for response-option row pattern
                option_vals = raw.iloc[i, 2:].tolist()
                option_labels = [
                    str(x).strip() for x in option_vals
                    if pd.notna(x) and str(x).strip() != ""
                ]

                if option_labels and any("(" in x and ")" in x for x in option_labels):
                    count_row = raw.iloc[i + 1, 2:2 + len(option_labels)].tolist() if i + 1 < len(raw) else []
                    share_row = raw.iloc[i + 2, 2:2 + len(option_labels)].tolist() if i + 2 < len(raw) else []

                    for j, opt in enumerate(option_labels):
                        count_val = count_row[j] if j < len(count_row) else None
                        share_val = share_row[j] if j < len(share_row) else None

                        records.append({
                            "Year": year,
                            "SourceFile": fp.name,
                            "QuestionText": qtext,
                            "ResponseOption": opt,
                            "Count": count_val,
                            "Share": share_val
                        })

                    i += 3
                    continue

        i += 1

    out = pd.DataFrame(records)
    return out


def combine_years(file_paths: list[str]) -> pd.DataFrame:
    frames = []
    for fp in file_paths:
        frames.append(parse_summary_sheet(fp))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    files = [
        # r"2021-2023/SA EXIT QUESTIONNAIRE Fall 2021.xlsx",
        # r"2021-2023/SA EXIT QUESTIONNAIRE Fall 2022.xlsx",
        # r"2021-2023/SA EXIT QUESTIONNAIRE Spring 2023.xlsx",
        r"",
    ]

    master = combine_years(files)
    master.to_csv("sa_exit_summary_2021_2023.csv", index=False)
    master.to_parquet("sa_exit_summary_2021_2023.parquet", index=False)

    print("Wrote:", "sa_exit_summary_2021_2023.csv", "and", "sa_exit_summary_2021_2023.parquet")