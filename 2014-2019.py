# =========================
# 2014-2019.py  (FULL SCRIPT)
# =========================
# sa_exit_2016format_master.py
# Purpose: Process ONLY the "2016-style" SA EXIT QUESTIONNAIRE files into the SAME long-format output schema.
# - Questions sheet: Q IDs live in a column like "Unnamed: 0" (values like Q1, Q2, ...) and question text is in "Question"
# - Section headers: blank Q ID rows where the text column contains the section title
# - Answers sheet: contains columns like Q10c/Q10d/etc and/or Survey Data1/Survey Data2, etc.
#
# Output (per response row): metadata cols from Answers + Year/SourceFile/ResponseID + QuestionRaw/ResponseText +
#                           QuestionNumber/ResponseType + QuestionText/Section
#
# This script writes outputs to a separate folder so you can keep formats separate for now.

import re
from pathlib import Path
from typing import Optional, List

import pandas as pd


# -----------------------------
# 0) CONFIG (edit if you want)
# -----------------------------
OUTPUT_DIR = Path("NU_FINAL")
OUTPUT_DIR.mkdir(exist_ok=True)


def pick_sheet(xls: pd.ExcelFile, candidates) -> Optional[str]:
    """
    Return the actual sheet name from xls that matches any candidate,
    case-insensitive, ignoring whitespace.
    """
    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s).lower()

    sheet_map = {norm(s): s for s in xls.sheet_names}
    for c in candidates:
        key = norm(c)
        if key in sheet_map:
            return sheet_map[key]
    return None


# -----------------------------
# 1) YEAR inference
# -----------------------------
def infer_year_from_filename(fp: Path) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", fp.name)
    return int(m.group()) if m else None


def infer_year_from_any_datecol(df: pd.DataFrame) -> Optional[int]:
    for col in ["Date Submitted", "Date submitted", "Created", "Created Date", "Timestamp"]:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().any():
                return int(dt.dropna().iloc[0].year)
    return None


# -----------------------------
# 2) Questions sheet parsing (2016-style)
# -----------------------------
def normalize_question_id(x) -> Optional[str]:
    """
    Normalize IDs from Questions sheet so we can join:

      Q12, Q12a, Q12dd -> Q12
      Survey Data4     -> Survey Data4
      Survey Data4a    -> Survey Data4a
      Multiline7       -> Multiline7
      Multiline1a      -> Multiline1a

    Anything else is returned as its stripped value.
    """
    if pd.isna(x):
        return None
    s = str(x).strip()

    # Q variants -> Q<number>
    m = re.match(r"^Q\s*(\d+)", s, flags=re.I)
    if m:
        return f"Q{int(m.group(1))}"

    # Survey Data variants -> Survey Data<number>(optional suffix)
    m = re.match(r"^Survey\s*Data\s*(\d+)([a-z]+)?$", s, flags=re.I)
    if m:
        suf = (m.group(2) or "")
        return f"Survey Data{int(m.group(1))}{suf}"

    # Multiline variants -> Multiline<number>(optional letter)
    m = re.match(r"^Multiline\s*(\d+)([a-z])?$", s, flags=re.I)
    if m:
        letter = (m.group(2) or "").lower()
        return f"Multiline{int(m.group(1))}{letter}"

    return s


def pick_qid_col_2016(questions_df: pd.DataFrame) -> str:
    """
    In 2016-style, the QID column is typically 'Unnamed: 0' but we infer it robustly:
    choose the column with the most entries matching Q<number> OR Survey Data<number> OR Multiline<number>.
    """
    id_like_counts = {}
    for c in questions_df.columns:
        s = questions_df[c].astype(str)

        count_q = s.str.match(r"^\s*Q\d+([a-z]+)?\s*$", case=False, na=False).sum()
        count_sd = s.str.match(r"^\s*Survey\s*Data\s*\d+([a-z]+)?\s*$", case=False, na=False).sum()
        count_ml = s.str.match(r"^\s*Multiline\s*\d+([a-z])?\s*$", case=False, na=False).sum()

        id_like_counts[c] = int(count_q + count_sd + count_ml)

    best = max(id_like_counts, key=id_like_counts.get)
    if id_like_counts[best] == 0:
        raise ValueError(
            "Could not find an ID column in Questions sheet (expected many values like Q1, Survey Data1, Multiline7). "
            f"Columns found: {list(questions_df.columns)}"
        )
    return best


def pick_text_col_2016(questions_df: pd.DataFrame, qid_col: str) -> str:
    """
    Prefer 'Question' if present; otherwise pick the non-qid column with the most non-null entries.
    """
    if "Question" in questions_df.columns:
        return "Question"
    candidates = [c for c in questions_df.columns if c != qid_col]
    if not candidates:
        raise ValueError("Questions sheet has no candidate text column besides the ID column.")
    return max(candidates, key=lambda c: questions_df[c].notna().sum())


def build_questions_dim_2016(questions_df: pd.DataFrame) -> pd.DataFrame:
    qid_col = pick_qid_col_2016(questions_df)
    text_col = pick_text_col_2016(questions_df, qid_col)

    q = questions_df[[qid_col, text_col]].copy()
    q = q.rename(columns={qid_col: "RawID", text_col: "RawText"})

    # Section headers: RawID is NaN; RawText contains the section title
    q["Section"] = q["RawText"].where(q["RawID"].isna())
    q["Section"] = q["Section"].ffill()

    # Actual questions (Q, Survey Data, Multiline)
    q["QuestionNumber"] = q["RawID"].apply(normalize_question_id)
    q = q[q["QuestionNumber"].notna()].copy()

    q["QuestionText"] = q["RawText"].astype(str).str.strip()
    return q[["QuestionNumber", "QuestionText", "Section"]]


# -----------------------------
# 3) Answers sheet parsing
# -----------------------------
def is_question_col_2016(colname: str) -> bool:
    """
    Treat these as "question" columns to melt:
      - Q10, Q10a, Q10c, Q10dd, etc.
      - Survey Data1, Survey Data2, ...
      - Multiline1, Multiline2..., Multiline1a...
    """
    if not isinstance(colname, str):
        return False
    c = colname.strip()

    if re.match(r"^Q\d+([a-z]+)?$", c, flags=re.I):
        return True
    if re.match(r"^Survey\s*Data\d+([a-z]+)?$", c, flags=re.I):
        return True
    if re.match(r"^Multiline\d+([a-z]+)?$", c, flags=re.I):
        return True
    return False


def qraw_to_qnum(qraw: str) -> Optional[str]:
    """
    Normalize Answer sheet column names into the same join key as Questions sheet.
      Q12a -> Q12
      Survey Data4 -> Survey Data4
      Multiline7 -> Multiline7
      Multiline1a -> Multiline1a
    """
    if not isinstance(qraw, str):
        return None
    s = qraw.strip()

    m = re.match(r"^Q\s*(\d+)", s, flags=re.I)
    if m:
        return f"Q{int(m.group(1))}"

    m = re.match(r"^Survey\s*Data\s*(\d+)([a-z]+)?$", s, flags=re.I)
    if m:
        suf = (m.group(2) or "")
        return f"Survey Data{int(m.group(1))}{suf}"

    m = re.match(r"^Multiline\s*(\d+)([a-z])?$", s, flags=re.I)
    if m:
        letter = (m.group(2) or "").lower()
        return f"Multiline{int(m.group(1))}{letter}"

    return None


def classify_response_type_2016(question_raw: str, response_text: str) -> str:
    qr = (question_raw or "").strip()
    rt = (response_text or "").strip()

    # long text buckets
    if re.match(r"^(Survey\s*Data|Multiline)", qr, flags=re.I):
        return "LongText"

    # Likert is often "(2) Agree"
    if re.match(r"^\(\s*\d+\s*\)", rt):
        return "Likert"

    return "Comment"


def process_one_2016_file(excel_path: str) -> pd.DataFrame:
    fp = Path(excel_path)
    xls = pd.ExcelFile(fp)

    q_sheet = pick_sheet(xls, ["Questions"])
    a_sheet = pick_sheet(xls, ["Answers", "Response", "Responses"])

    if q_sheet is None or a_sheet is None:
        raise ValueError(
            f"{fp.name} must contain a Questions sheet and an Answers/Responses sheet. "
            f"Found sheets: {xls.sheet_names}"
        )

    questions = pd.read_excel(xls, q_sheet)
    answers = pd.read_excel(xls, a_sheet)

    # Drop Excel index/junk columns like "Unnamed: 0"
    answers = answers.loc[:, ~answers.columns.astype(str).str.match(r"^Unnamed:", case=False)]
    # Clean column names (helps matching Q / Survey Data columns)
    answers = answers.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    qdim = build_questions_dim_2016(questions)

    year = infer_year_from_filename(fp) or infer_year_from_any_datecol(answers)

    a = answers.copy().reset_index(drop=True)
    a["Year"] = year
    a["SourceFile"] = fp.name
    a["ResponseID"] = [f"{fp.stem}-{i+1}" for i in range(len(a))]

    value_cols = [c for c in a.columns if is_question_col_2016(c)]
    id_cols = [c for c in a.columns if c not in value_cols]

    long = a.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="QuestionRaw",
        value_name="ResponseText",
    )

    # Drop blanks
    long = long.dropna(subset=["ResponseText"]).copy()
    long["ResponseText"] = long["ResponseText"].astype(str).str.strip()
    long = long[long["ResponseText"] != ""]

    # QuestionNumber for joining
    long["QuestionNumber"] = long["QuestionRaw"].apply(qraw_to_qnum)

    # Type
    long["ResponseType"] = [
        classify_response_type_2016(q, r)
        for q, r in zip(long["QuestionRaw"], long["ResponseText"])
    ]

    # Merge question text + section (NOW Survey Data + Multiline WILL JOIN)
    long = long.merge(qdim, on="QuestionNumber", how="left")
    long["QuestionText"] = long["QuestionText"].fillna(long["QuestionRaw"])
    long["Section"] = long["Section"].fillna("")

    return long


def combine_years_2016format(files: List[str]) -> pd.DataFrame:
    frames = []
    for f in files:
        frames.append(process_one_2016_file(f))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# -----------------------------
# 4) Optional: align column order to an existing schema
# -----------------------------
def align_to_schema(df_long: pd.DataFrame, schema_csv_path: str) -> pd.DataFrame:
    schema_cols = list(pd.read_csv(schema_csv_path, nrows=1).columns)
    out = df_long.copy()
    for c in schema_cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[schema_cols]


# -----------------------------
# 5) MAIN
# -----------------------------
if __name__ == "__main__":
    files = [
        r"NuIN/NU EXIT QUESTIONNAIRE Fall 2012.xlsx",
        r"NuIN/NUin Exit Questionnaire Fall 2008.xls",
        r"NuIN/NU EXIT QUESTIONNAIRE Fall 2014.xls",
        r'NuIN/NU EXIT QUESTIONNAIRE Fall 2016.xlsx',
        r'NuIN/NU EXIT QUESTIONNAIRE Fall 2017.xlsx',
        r'NuIN/NU EXIT QUESTIONNAIRE Fall 2018.xlsx'
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE Fall 2016.xlsx",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE Fall 2017.xlsx",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE Fall 2018.xlsx",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE Fall 2019.xlsx",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE Spring 2017.xlsx",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE Spring 2018.xlsx",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE Spring 2019.xlsx",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE Summer 2017.xlsx",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE Summer 2018.xlsx",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE Summer 2019.xlsx",
        # add more 2016-format files here...
    ]

    master = combine_years_2016format(files)

    # OPTIONAL: if you want the EXACT column order of your ideal master, uncomment and point to it.
    # master = align_to_schema(master, r"sa_exit_master_long.csv")

    out_csv = OUTPUT_DIR / "NU_exit_master_long_2016format.csv"
    #out_parquet = OUTPUT_DIR / "sa_exit_master_long_2016format.parquet"

    # Safety net: remove any leftover "Unnamed:*" columns before saving
    master = master.loc[:, ~master.columns.astype(str).str.match(r"^Unnamed:", case=False)]

    master.to_csv(out_csv, index=False)

    # Make Age numeric (strings like "20" -> 20.0). Non-numeric -> NaN
    if "Age" in master.columns:
        master["Age"] = pd.to_numeric(master["Age"], errors="coerce")

    #master.to_parquet(out_parquet, index=False)

    print("Wrote:", str(out_csv))


