
import pandas as pd
import re

# ============================================================
# FILE 1: extract_trip_comments_from_master.py
# Purpose:
#   Start with the merged ACT master dataset and create a clean
#   trip / excursion comments file for later analysis.
#
# Input:
#   sa_exit_MASTER_merged_all4.csv
#
# Output:
#   trip_comments_strict_places_filter.csv
# ============================================================

MASTER_PATH = "sa_exit_MASTER_merged_all4.csv"
OUT_PATH = "trip_comments_strict_places_filter.csv"

df = pd.read_csv(MASTER_PATH, low_memory=False)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

for col in ["QuestionText", "QuestionRaw", "ResponseType", "ResponseText"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].astype(str)

# keep only real text responses
df = df[df["ResponseText"].notna()].copy()
df = df[~df["ResponseText"].str.strip().isin(["", "nan", "None", "N/A", "n/a"])].copy()

# comments only
df = df[df["ResponseType"].str.contains("Comment|LongText", case=False, na=False)].copy()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("&amp;", "&").replace("&#039;", "'")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s'\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_comment"] = df["ResponseText"].apply(clean_text)

trip_words = [
    "field trip", "field trips", "trip", "trips", "excursion", "excursions"
]

greek_places = [
    "athens", "delphi", "vergina", "meteora", "olympus", "mount olympus",
    "pozar", "ioannina", "kavala", "philippi", "crete",
    "corfu", "santorini", "mykonos", "nafplio", "sparta", "epidaurus",
    "mycenae", "parthenon", "acropolis", "white tower", "rotunda",
    "arch of galerius", "monasteries", "museum", "museums", "halkidiki"
]

academic_words = [
    "class", "classes", "course", "courses", "professor", "professors",
    "teacher", "teachers", "semester", "content", "psychology",
    "syllabus", "lecture", "lectures", "homework", "exam", "exams",
    "academic", "academics"
]

non_trip_words = [
    "intramurals", "clubs", "wifi", "housing", "apartment", "dorm",
    "roommate", "residence life", "campus", "weekend activities"
]

def is_trip_comment(text: str) -> bool:
    text = clean_text(text)

    has_trip_word = any(term in text for term in trip_words)
    has_place_word = any(term in text for term in greek_places)

    has_academic = any(term in text for term in academic_words)
    has_non_trip = any(term in text for term in non_trip_words)

    if not (has_trip_word or has_place_word):
        return False

    # avoid academic comments that name a place but are not about excursions
    if has_academic and not has_trip_word:
        return False

    # avoid non-trip operational comments that mention a place
    if has_non_trip and not has_trip_word:
        return False

    return True

trip_df = df[df["clean_comment"].apply(is_trip_comment)].copy()

# remove exact duplicate comments within the same year
trip_df = trip_df.drop_duplicates(subset=["Year", "ResponseText"]).copy()

out = trip_df[["Year", "ResponseText"]].rename(
    columns={"Year": "year", "ResponseText": "comment"}
).copy()

out.to_csv(OUT_PATH, index=False)

print(f"Wrote {OUT_PATH} with {len(out):,} trip-related comments.")
