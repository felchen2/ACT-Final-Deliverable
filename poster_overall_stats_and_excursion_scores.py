
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ============================================================
# FILE 3: poster_overall_stats_and_excursion_scores.py
# Purpose:
#   Recreate the poster/report quantitative visuals tied to:
#   - Overall Stats panel on the poster
#   - Excursions score-over-time chart
#
# Inputs:
#   sa_exit_MASTER_merged_all4.csv
#
# Outputs:
#   overall_satisfaction_by_year.png
#   share_positive_responses_over_time.png
#   excursion_rating_by_year.png
# ============================================================

MASTER_PATH = "sa_exit_MASTER_merged_all4.csv"

df = pd.read_csv(MASTER_PATH, low_memory=False)

for col in ["QuestionText", "ResponseText", "SourceFile"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Likert reverse mapping: 5 = best, 1 = worst
likert_map = {
    "(1) Strongly Agree": 5,
    "(2) Agree": 4,
    "(3) Neutral": 3,
    "(4) Disagree": 2,
    "(5) Strongly Disagree": 1,
    "1": 5, "2": 4, "3": 3, "4": 2, "5": 1
}

df["score"] = df["ResponseText"].map(likert_map)

# term extracted from SourceFile
def extract_term(s):
    s = str(s).lower()
    if "fall" in s:
        return "Fall"
    if "spring" in s:
        return "Spring"
    if "summer" in s:
        return "Summer"
    return np.nan

df["Term"] = df["SourceFile"].apply(extract_term)

# 1) Overall satisfaction by year
overall_mask = df["QuestionText"].str.contains(
    r"worthwhile study abroad experience|overall satisfaction|overall experience|recommend",
    case=False, regex=True
)
overall = (
    df.loc[overall_mask & df["score"].notna()]
      .groupby("Year", as_index=False)["score"]
      .mean()
      .sort_values("Year")
)
plt.figure(figsize=(10, 5.8))
plt.plot(overall["Year"], overall["score"], marker="o")
plt.title("Overall Satisfaction by Year (5 = Highest, 1 = Lowest)")
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.ylim(1, 5)
plt.yticks([1,2,3,4,5])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("overall_satisfaction_by_year.png", dpi=300)
plt.close()

# 2) Share of positive responses over time
# Use recommend + worthwhile style questions as the poster's high-level positive share.
positive_mask = df["QuestionText"].str.contains(
    r"recommend|worthwhile study abroad experience",
    case=False, regex=True
)
positive_df = df.loc[positive_mask & df["ResponseText"].ne("")].copy()
positive_df["is_positive"] = positive_df["ResponseText"].isin(["(1) Strongly Agree", "(2) Agree"])

share_pos = (
    positive_df.groupby("Year", as_index=False)
    .agg(pct_positive=("is_positive", lambda s: s.mean() * 100))
    .sort_values("Year")
)

plt.figure(figsize=(10, 5.8))
plt.plot(share_pos["Year"], share_pos["pct_positive"], marker="o", label="Positive response share")
plt.title("Share of Positive Responses Over Time")
plt.xlabel("Year")
plt.ylabel("Percent positive")
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("share_positive_responses_over_time.png", dpi=300)
plt.close()

# 3) Excursion ratings by year
excursion_mask = df["QuestionText"].str.contains(
    r"field trip|field trips|optional field trips|included field trips|excursion|excursions",
    case=False, regex=True
)
exc = (
    df.loc[excursion_mask & df["score"].notna()]
      .groupby("Year", as_index=False)["score"]
      .mean()
      .sort_values("Year")
)

plt.figure(figsize=(10, 5.8))
plt.plot(exc["Year"], exc["score"], marker="o")
plt.title("Excursions Rating by Year (5 = Highest, 1 = Lowest)")
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.ylim(1, 5)
plt.yticks([1,2,3,4,5])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("excursion_rating_by_year.png", dpi=300)
plt.close()

print("Saved poster/report quantitative charts.")
