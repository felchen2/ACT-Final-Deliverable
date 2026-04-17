import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ============================================================
# FILE 2: analyze_excursions_from_trip_comments.py
#
# Purpose:
#   Take the cleaned trip comment file and reproduce the core
#   excursion analysis used in the report/poster:
#   - remove bad multiple-choice text rows
#   - sentiment labels using VADER
#   - coded themes, including a staff category
#   - positive vs negative pie chart
#   - trip problem trends over time
#   - destination positivity
#   - coded CSV used for writeup examples
#
# Input:
#   trip_comments_strict_places_filter.csv
#
# Outputs:
#   trip_comments_cleaned_for_analysis.csv
#   trip_comments_removed_as_multiple_choice.csv
#   trip_comments_coded_for_report.csv
#   trip_comment_summary_stats.csv
#   trip_sentiment_distribution.png
#   excursion_comments_positive_vs_negative.png
#   trip_theme_counts.png
#   trip_problem_trends.png
#   trip_destination_positivity.png
#   trip_destination_positivity_table.csv
# ============================================================

INPUT_PATH = "trip_comments_strict_places_filter.csv"

# ------------------------------------------------------------
# VADER setup
# ------------------------------------------------------------
# First time only, you may need:
# import nltk
# nltk.download("vader_lexicon")

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["comment"] = df["comment"].astype(str)

# ------------------------------------------------------------
# Basic cleaning
# ------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = str(text)
    text = text.replace("&amp;", "&").replace("&#039;", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_for_matching(text: str) -> str:
    return normalize_text(text).lower()

df["comment"] = df["comment"].apply(normalize_text)
df["clean"] = df["comment"].apply(clean_for_matching)

# remove blank / useless rows
bad_empty_values = {"", "nan", "none", "n/a", "na"}
df = df[~df["clean"].isin(bad_empty_values)].copy()

# ------------------------------------------------------------
# Remove multiple-choice / checkbox-style responses
# ------------------------------------------------------------
def is_multiple_choice_response(text: str) -> bool:
    text = str(text).strip()

    # full line made only of repeated "(number) label" chunks
    pattern_full_mc = r"^\s*(\(\d+\)\s*[^()]+?)(\s+\(\d+\)\s*[^()]+?)*\s*$"
    if re.fullmatch(pattern_full_mc, text):
        return True

    # if 2+ option markers show up, probably checkbox export text
    option_markers = re.findall(r"\(\d+\)", text)
    if len(option_markers) >= 2:
        return True

    # single option exports that are clearly not real comments
    single_option_phrases = [
        "field trips (optional)",
        "sports (intramurals)",
        "sports (varsity)",
        "clubs",
        "other campus events"
    ]
    if text.lower().startswith("(") and any(p in text.lower() for p in single_option_phrases):
        return True

    return False

df["is_multiple_choice"] = df["comment"].apply(is_multiple_choice_response)

removed_mc = df[df["is_multiple_choice"]].copy()
removed_mc.to_csv("trip_comments_removed_as_multiple_choice.csv", index=False)

df = df[~df["is_multiple_choice"]].copy()
df = df.drop(columns=["is_multiple_choice"], errors="ignore")

# save cleaned file used for analysis
df.to_csv("trip_comments_cleaned_for_analysis.csv", index=False)

# ------------------------------------------------------------
# Sentiment using VADER
# ------------------------------------------------------------
def vader_scores(text: str) -> dict:
    return sia.polarity_scores(text)

df["vader_dict"] = df["comment"].apply(vader_scores)
df["sent_score"] = df["vader_dict"].apply(lambda d: d["compound"])
df["sent_pos"] = df["vader_dict"].apply(lambda d: d["pos"])
df["sent_neu"] = df["vader_dict"].apply(lambda d: d["neu"])
df["sent_neg"] = df["vader_dict"].apply(lambda d: d["neg"])

# Common VADER cutoffs
df["sent_label"] = np.where(
    df["sent_score"] >= 0.05,
    "Positive",
    np.where(df["sent_score"] <= -0.05, "Negative", "Neutral")
)

# ------------------------------------------------------------
# Theme categories
# ------------------------------------------------------------
cats = {
    "Positive overall / enjoyment": [
        r"\blove\b", r"\bloved\b", r"\benjoy", r"\bgreat\b", r"\bamazing\b",
        r"\bawesome\b", r"\bfavorite\b", r"\bbeautiful\b", r"\bgood\b",
        r"\bworthwhile\b", r"\bmemorable\b", r"\bfun\b"
    ],
    "Educational / cultural value": [
        r"informative", r"interesting", r"educational", r"history",
        r"historical", r"learned", r"cultural", r"informational",
        r"museum", r"museums", r"\bguide\b", r"tour guide"
    ],
    "Wanted more trips / more options": [
        r"more trips", r"more field trips", r"wish.*more", r"not enough",
        r"too few", r"more options", r"another trip", r"wanted more",
        r"plan more", r'another'
    ],
    "Organization / scheduling / communication": [
        r"well organized", r"\borganized\b", r"spread out", r"confusing",
        r"didn.?t know", r"communication", r"last minute", r"better informed",
        r"schedule", r"scheduling", r"timing", r"unclear", r"disorganized",
        r"poorly planned", r"planning"
    ],
    "Cost / affordability": [
        r"expensive", r"\bcost\b", r"paid", r"prepaid", r"afford",
        r"price", r"overpriced", r"costly"
    ],
    "Transportation / logistics": [
        r"\bbus\b", r"transport", r"\bride\b", r"\bfar\b", r"\bwalk\b",
        r"travel", r"location", r"free time", r"rushed", r"drive",
        r"driving"
    ],
    "Staff / staff support": [
        r"\bstaff\b", r"\bact staff\b", r"\bdirector\b", r"\bdirectors\b",
        r"\bprogram staff\b", r"\bcoordinator\b", r"\bcoordinators\b",
        r"\bleader\b", r"\bleaders\b", r"\bmentor\b", r"\bmentors\b",
        r"\bguide\b", r"\bguides\b", r"\bprofessor\b", r"\bprofessors\b",
        r"\bteacher\b", r"\bteachers\b", r"\bhelpful staff\b",
        r"\bsupportive staff\b", r"stepan", r'efi'
    ]
}

def assign_multi(text: str):
    matched = []
    for category, patterns in cats.items():
        if any(re.search(p, text) for p in patterns):
            matched.append(category)
    return matched if matched else ["Other"]

df["categories"] = df["clean"].apply(assign_multi)
ex = df.explode("categories").copy()

# ------------------------------------------------------------
# Save coded comment file
# ------------------------------------------------------------
coded = df[[
    "year",
    "comment",
    "sent_label",
    "sent_score",
    "sent_pos",
    "sent_neu",
    "sent_neg",
    "categories"
]].copy()

coded.to_csv("trip_comments_coded_for_report.csv", index=False)

# ------------------------------------------------------------
# Summary stats
# ------------------------------------------------------------
year_min = int(df["year"].dropna().min()) if df["year"].dropna().shape[0] > 0 else None
year_max = int(df["year"].dropna().max()) if df["year"].dropna().shape[0] > 0 else None

summary_stats = pd.DataFrame({
    "metric": [
        "Total trip-related comments after cleaning",
        "Rows removed as multiple choice",
        "Years covered",
        "Positive sentiment share",
        "Neutral sentiment share",
        "Negative sentiment share",
        "Average VADER compound score"
    ],
    "value": [
        len(df),
        len(removed_mc),
        f"{year_min}-{year_max}" if year_min is not None and year_max is not None else "Unknown",
        f"{(df['sent_label'].eq('Positive').mean() * 100):.1f}%",
        f"{(df['sent_label'].eq('Neutral').mean() * 100):.1f}%",
        f"{(df['sent_label'].eq('Negative').mean() * 100):.1f}%",
        round(df["sent_score"].mean(), 3)
    ]
})
summary_stats.to_csv("trip_comment_summary_stats.csv", index=False)

# ------------------------------------------------------------
# Chart 1: overall sentiment counts
# ------------------------------------------------------------
sent_counts = (
    df["sent_label"]
    .value_counts()
    .reindex(["Positive", "Neutral", "Negative"])
    .fillna(0)
)

plt.figure(figsize=(8, 5))
plt.bar(sent_counts.index, sent_counts.values)
plt.title("Overall Sentiment of Trip Comments")
plt.xlabel("Sentiment")
plt.ylabel("Number of comments")
plt.tight_layout()
plt.savefig("trip_sentiment_distribution.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Chart 2: positive vs negative pie chart
# ------------------------------------------------------------
pie_counts = (
    df["sent_label"]
    .value_counts()
    .reindex(["Positive", "Negative"])
    .fillna(0)
)

plt.figure(figsize=(7, 7))
plt.pie(
    pie_counts.values,
    labels=pie_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Excursion Comments: Positive vs Negative")
plt.tight_layout()
plt.savefig("excursion_comments_positive_vs_negative.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Chart 3: theme counts
# ------------------------------------------------------------
cat_counts = ex["categories"].value_counts().drop(labels=["Other"], errors="ignore")

plt.figure(figsize=(10, 6))
plt.bar(cat_counts.index, cat_counts.values)
plt.title("Main Themes in Trip Comments")
plt.xlabel("Theme")
plt.ylabel("Number of coded mentions")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("trip_theme_counts.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Chart 4: trip problem trends over time
# ------------------------------------------------------------
problem_themes = [
    "Cost / affordability",
    "Organization / scheduling / communication",
    "Transportation / logistics",
    "Wanted more trips / more options",
    "Staff / staff support"
]

trend = (
    ex[ex["categories"].isin(problem_themes)]
    .groupby(["year", "categories"])
    .size()
    .reset_index(name="count")
    .pivot(index="year", columns="categories", values="count")
    .fillna(0)
)

if not trend.empty:
    plt.figure(figsize=(11, 6))
    for col in trend.columns:
        plt.plot(trend.index, trend[col], marker="o", label=col)
    plt.title("How Common Excursion Issues Changed Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of coded mentions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("trip_problem_trends.png", dpi=300)
    plt.close()

# ------------------------------------------------------------
# Chart 5: destination positivity
# ------------------------------------------------------------
places = {
    "Athens": ["athens"],
    "Delphi": ["delphi"],
    "Vergina": ["vergina", "verginia"],
    "Meteora": ["meteora"],
    "Olympus": ["olympus", "mount olympus"],
    "Corfu": ["corfu"],
    "Pozar": ["pozar"]
}

rows = []
for name, kws in places.items():
    mask = df["clean"].apply(lambda t: any(k in t for k in kws))
    sub = df[mask].copy()

    if len(sub) >= 8:
        rows.append({
            "trip": name,
            "mentions": len(sub),
            "pct_positive": sub["sent_label"].eq("Positive").mean() * 100,
            "avg_vader_score": round(sub["sent_score"].mean(), 3)
        })

place_df = pd.DataFrame(rows)

if not place_df.empty:
    place_df = place_df.sort_values("pct_positive", ascending=False)
    place_df.to_csv("trip_destination_positivity_table.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(place_df["trip"], place_df["pct_positive"])
    plt.title("Share of Positive Comments by Named Trip")
    plt.xlabel("Trip")
    plt.ylabel("Positive comments (%)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig("trip_destination_positivity.png", dpi=300)
    plt.close()
else:
    pd.DataFrame(columns=["trip", "mentions", "pct_positive", "avg_vader_score"]).to_csv(
        "trip_destination_positivity_table.csv", index=False
    )

print("Saved excursion analysis outputs.")
print(f"Comments kept for analysis: {len(df):,}")
print(f"Multiple-choice rows removed: {len(removed_mc):,}")