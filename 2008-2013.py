# =========================
# 2008-2013.py  (FULL SCRIPT)
# =========================
import re
from pathlib import Path
import pandas as pd


def infer_year_from_filename(fp: Path) -> int | None:
    m = re.search(r"(19|20)\d{2}", fp.name)
    return int(m.group()) if m else None


def normalize_qnum(x) -> str | None:
    """
    Normalizes question numbers from Questions sheet:
      1   -> Q1
      Q1  -> Q1
      '  Q12 ' -> Q12
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    # if numeric-like
    if re.fullmatch(r"\d+", s):
        return f"Q{int(s)}"
    # if already Q#
    m = re.match(r"^Q\s*(\d+)$", s, flags=re.I)
    if m:
        return f"Q{int(m.group(1))}"
    return s  # fallback


def qraw_to_qnum(qraw: str) -> str | None:
    """
    Q12a -> Q12
    Q12b -> Q12
    Multiline1a -> None (won't match)
    """
    if not isinstance(qraw, str):
        return None
    m = re.match(r"^Q(\d+)", qraw.strip(), flags=re.I)
    return f"Q{int(m.group(1))}" if m else None


def build_questions_dim(questions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns in Questions sheet: Number, Description
    Also has section header rows where Number is blank.
    Output: QuestionNumber, QuestionText, Section
    """
    q = questions_df.copy()

    # Identify section header rows (Number is blank); use Description as the section name
    q["Section"] = q["Description"].where(q["Number"].isna())
    q["Section"] = q["Section"].ffill()

    # Keep actual questions only
    q = q[q["Number"].notna()].copy()

    q["QuestionNumber"] = q["Number"].apply(normalize_qnum)
    q["QuestionText"] = q["Description"].astype(str).str.strip()

    return q[["QuestionNumber", "QuestionText", "Section"]]


# -----------------------------
# NEW: Friendly labels for old-format Multiline blocks (2009–2013)
# -----------------------------
MULTILINE_PARENT = {
    1: "What were the major strengths of this program?",
    2: "What were the major weaknesses of this program, if any?",
    3: "What suggestions do you have for improving this program?",
}
MULTILINE_TOPIC = {
    "a": "ACADEMICS",
    "b": "STUDENT SERVICES",
    "c": "ACTIVITIES",
}
MULTILINE7_TEXT = "Do you have any additional comments or suggestions that were not covered above?"


def multiline_question_label(qraw: str) -> str | None:
    """
    Multiline1a -> 'What were the major strengths of this program? — ACADEMICS'
    Multiline2b -> 'What were the major weaknesses... — STUDENT SERVICES'
    Multiline7  -> 'Do you have any additional comments...'
    """
    if not isinstance(qraw, str):
        return None
    s = qraw.strip()

    m = re.match(r"^Multiline\s*(\d+)\s*([a-z])?$", s, flags=re.I)
    if not m:
        return None

    num = int(m.group(1))
    letter = (m.group(2) or "").lower()

    if num == 7:
        return MULTILINE7_TEXT

    parent = MULTILINE_PARENT.get(num)
    if not parent:
        return None

    topic = MULTILINE_TOPIC.get(letter)
    if topic:
        return f"{parent} — {topic}"
    return parent


def merge_questions_answers(file_path: str) -> pd.DataFrame:
    """
    For a single 'old format' SA Exit file:
    - reads Questions + Answers
    - melts Answers into long format
    - merges on QuestionNumber
    """
    fp = Path(file_path)
    year = infer_year_from_filename(fp)

    xls = pd.ExcelFile(fp)
    if not {"Questions", "Answers"}.issubset(set(xls.sheet_names)):
        raise ValueError(f"{fp.name} does not have both 'Questions' and 'Answers' sheets")

    questions = pd.read_excel(xls, "Questions")
    answers = pd.read_excel(xls, "Answers")

    qdim = build_questions_dim(questions)

    a = answers.copy().reset_index(drop=True)
    a["Year"] = year
    a["SourceFile"] = fp.name
    a["ResponseID"] = [f"{fp.stem}-{i+1}" for i in range(len(a))]

    # Identify question columns (typical: Q1a/Q1b/... plus Multiline*)
    q_cols = [c for c in a.columns if isinstance(c, str) and re.match(r"^Q\d+[ab]$", c)]
    multiline_cols = [c for c in a.columns if isinstance(c, str) and re.match(r"^Multiline\d+", c, flags=re.I)]
    value_cols = q_cols + multiline_cols

    # Everything else is treated as respondent metadata
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

    # Derive QuestionNumber for joining (Q12a -> Q12)
    long["QuestionNumber"] = long["QuestionRaw"].apply(qraw_to_qnum)

    # Optional: classify response type
    def response_type(qraw: str) -> str:
        if re.match(r"^Q\d+a$", qraw or ""):
            return "Likert"
        if re.match(r"^Q\d+b$", qraw or ""):
            return "Comment"
        if re.match(r"^Multiline", qraw or "", flags=re.I):
            return "LongText"
        return "Other"

    long["ResponseType"] = long["QuestionRaw"].apply(response_type)

    # Merge in QuestionText + Section
    long = long.merge(qdim, on="QuestionNumber", how="left")

    # -----------------------------
    # NEW: Fill Multiline question prompts + topics (2009–2013)
    # -----------------------------
    ml_mask = long["QuestionRaw"].astype(str).str.match(r"^Multiline", case=False, na=False)
    ml_labels = long.loc[ml_mask, "QuestionRaw"].map(multiline_question_label)

    # For multiline, set Section to COMMENTS (if missing) and set QuestionText to nice label (if missing)
    long.loc[ml_mask, "Section"] = long.loc[ml_mask, "Section"].fillna("COMMENTS")
    long.loc[ml_mask, "QuestionText"] = long.loc[ml_mask, "QuestionText"].fillna(ml_labels)

    # Final fallback if still missing
    long["QuestionText"] = long["QuestionText"].fillna(long["QuestionRaw"])
    long["Section"] = long["Section"].fillna("")

    return long


def combine_years(file_paths: list[str]) -> pd.DataFrame:
    frames = []
    for fp in file_paths:
        frames.append(merge_questions_answers(fp))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    files = [
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE- FALL 2009.xlsx",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE- SPRING 2010.xlsx",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE- FALL 2010.XLS",
        #r"2009-2012/Evals/SA_EXIT_QUESTIONNAIRE-_SPRING_2011(1).XLS",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE- SUMMER 2011.xls",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE Fall 2012.xlsx",
        #r"2009-2012/Evals/SA EXIT QUESTIONNAIRE Spring 2013.xlsx",
        r'',
        r'',
        r'',
        r'',
        # add more 2009–2013 style files here...
    ]

    master = combine_years(files)

    master.to_csv("sa_exit_master_long.csv", index=False)
    master.to_parquet("sa_exit_master_long.parquet", index=False)

    print("Wrote:", "sa_exit_master_long.csv", "and", "sa_exit_master_long.parquet")
