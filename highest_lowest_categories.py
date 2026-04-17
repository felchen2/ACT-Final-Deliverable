"""
SA Exit Survey Analysis
========================
Analyzes both Likert (quantitative) and open-ended (qualitative) survey
responses across Year, Class Level (proxy for term/age cohort), and
Student Status (cohort type).

MODEL CHOICE — Why TF-IDF + NMF?
----------------------------------
We use Non-negative Matrix Factorization (NMF) on TF-IDF vectors as our
"language model" for the qualitative text. This is a deliberate, bias-
conscious choice:

1. NO PRETRAINED BIASES: Large language models (BERT, GPT, etc.) are
   pretrained on massive internet corpora that encode societal biases
   (gender, race, political framing). NMF on TF-IDF is trained ONLY on
   your own survey text — it cannot inherit outside biases.

2. INTERPRETABLE TOPICS: NMF produces human-readable topic keywords.
   You can audit exactly what the model found. Black-box LLMs cannot
   offer this transparency.

3. REPRODUCIBLE: With a fixed random seed, every run produces identical
   results. Neural embedding models can shift across versions.

4. LIGHTWEIGHT & OFFLINE: Runs entirely on local data, no API keys,
   no third-party servers receiving your students' private responses.

5. NO SENTIMENT LEXICON BIAS: Pre-built sentiment libraries (VADER,
   TextBlob) embed assumptions about what words are "positive" or
   "negative." Instead we let the data reveal its own themes.

The one trade-off: NMF cannot understand context as deeply as a neural
model. For a small-to-medium survey corpus this is fine — the themes
it finds are stable and directly tied to the actual words respondents used.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe in scripts)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
# low_memory=False avoids dtype-inference warnings on mixed-type columns.

df = pd.read_csv("sa_exit_MASTER_merged_all4.csv", low_memory=False)

print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CLEAN & STANDARDISE DEMOGRAPHIC COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
# The raw columns carry ordinal prefixes like "(3) Junior".
# We strip the numeric prefix so groupbys are readable.

def strip_prefix(s):
    """Remove leading '(N) ' from a string, e.g. '(3) Junior' → 'Junior'."""
    if pd.isna(s):
        return np.nan
    return re.sub(r"^\(\d+\)\s*", "", str(s)).strip()

df["class_level"] = df["Class Level"].apply(strip_prefix)
df["student_status"] = df["Student Status"].apply(strip_prefix)
df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

# Extract term (Fall / Spring / Summer) from the SourceFile name.
# This is more reliable than parsing 'Date Submitted' across inconsistent formats.
def extract_term(src):
    if pd.isna(src):
        return "Unknown"
    s = str(src).lower()
    if "fall" in s:
        return "Fall"
    if "spring" in s:
        return "Spring"
    if "summer" in s:
        return "Summer"
    return "Unknown"

df["term"] = df["SourceFile"].apply(extract_term)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SPLIT INTO LIKERT vs. QUALITATIVE RESPONSES
# ─────────────────────────────────────────────────────────────────────────────
# ResponseType tells us what kind of answer each row contains.
# 'Likert'      → numeric scores we can average.
# 'Comment' / 'LongText' → free-text we analyse with NMF.

LIKERT_TYPES   = {"Likert"}
QUALITATIVE_TYPES = {"Comment", "LongText"}

likert_df = df[df["ResponseType"].isin(LIKERT_TYPES)].copy()
qual_df   = df[df["ResponseType"].isin(QUALITATIVE_TYPES)].copy()

print(f"\nLikert rows  : {len(likert_df):,}")
print(f"Qualitative rows: {len(qual_df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PARSE LIKERT SCORES TO NUMBERS
# ─────────────────────────────────────────────────────────────────────────────
# Responses look like '(1) Strongly Agree' … '(5) Strongly Disagree'.
# We extract the leading digit and REVERSE the scale so that:
#   5 = Strongly Agree (best)   1 = Strongly Disagree (worst)
# This makes higher scores mean "more positive", which is intuitive.
# "Not Applicable" rows are dropped because they carry no sentiment signal.

def parse_likert(text):
    """Return float 1–5 (reversed) or NaN."""
    if pd.isna(text):
        return np.nan
    m = re.match(r"^\((\d)\)", str(text).strip())
    if not m:
        return np.nan
    raw = int(m.group(1))
    if raw in (0, 6):          # Not Applicable variants
        return np.nan
    # Original scale: 1=SA, 2=A, 3=N, 4=D, 5=SD
    # Reversed:       5=SA, 4=A, 3=N, 2=D, 1=SD
    reversed_score = 6 - raw
    if reversed_score < 1 or reversed_score > 5:
        return np.nan
    return float(reversed_score)

likert_df["score"] = likert_df["ResponseText"].apply(parse_likert)
likert_df = likert_df.dropna(subset=["score"])

# Keep only the clean Section labels that represent meaningful categories.
CLEAN_SECTIONS = {
    "ACADEMIC PROGRAM", "STUDENT SERVICES", "ACTIVITIES",
    "OVERALL ASSESSMENT", "INTEGRATION AND SAFETY", "WELLNESS SERVICES",
}
likert_clean = likert_df[likert_df["Section"].isin(CLEAN_SECTIONS)].copy()

print(f"\nLikert rows after cleaning: {len(likert_clean):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  HELPER: consistent plot style
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = "Set2"
sns.set_theme(style="whitegrid", font_scale=1.1)

# def save_fig(name):
#     plt.tight_layout()
#     path = f"/mnt/user-data/outputs/{name}.png"
#     # plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  LIKERT ANALYSIS A — Overall highest & lowest category across all years
# ─────────────────────────────────────────────────────────────────────────────
# WHY: The Section column groups questions into thematic areas. Averaging
# all scores within each section gives a summary rating comparable across
# categories, years, demographics — free of individual question wording bias.

print("\n=== A. Overall category scores (all years) ===")
cat_overall = (
    likert_clean
    .groupby("Section")["score"]
    .agg(mean_score="mean", n="count", sem=lambda x: x.std() / np.sqrt(len(x)))
    .reset_index()
    .sort_values("mean_score", ascending=False)
)
print(cat_overall.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
colors = sns.color_palette(PALETTE, len(cat_overall))
bars = ax.barh(cat_overall["Section"], cat_overall["mean_score"], color=colors)
ax.errorbar(
    cat_overall["mean_score"], range(len(cat_overall)),
    xerr=cat_overall["sem"] * 1.96,   # 95 % CI
    fmt="none", color="black", linewidth=1.5, capsize=4,
)
ax.axvline(3, color="grey", linestyle="--", linewidth=1, label="Neutral (3.0)")
ax.set_xlabel("Mean Score (1=Strongly Disagree, 5=Strongly Agree)")
ax.set_title("Overall Mean Score by Category (All Years)\n95% CI shown")
ax.set_xlim(1, 5)
ax.legend()
# save_fig("A_overall_category_scores")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  LIKERT ANALYSIS B — Category scores by Year (trend)
# ─────────────────────────────────────────────────────────────────────────────
# WHY: Plotting the mean per Section × Year shows whether satisfaction in
# a category improved, declined, or stayed flat over time — a key question
# for longitudinal programme evaluation.

print("\n=== B. Category scores by Year ===")
cat_year = (
    likert_clean
    .groupby(["year", "Section"])["score"]
    .mean()
    .reset_index()
    .rename(columns={"score": "mean_score"})
)

fig, ax = plt.subplots(figsize=(12, 6))
sections = cat_year["Section"].unique()
palette  = sns.color_palette(PALETTE, len(sections))
for sec, col in zip(sections, palette):
    sub = cat_year[cat_year["Section"] == sec].sort_values("year")
    ax.plot(sub["year"], sub["mean_score"], marker="o", label=sec, color=col)
ax.axhline(3, color="grey", linestyle="--", linewidth=1, label="Neutral")
ax.set_xlabel("Academic Year")
ax.set_ylabel("Mean Score (1–5)")
ax.set_title("Category Scores by Year")
ax.legend(loc="lower right", fontsize=9)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
# save_fig("B_category_by_year")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  LIKERT ANALYSIS C — Category scores by Class Level (age proxy / term)
# ─────────────────────────────────────────────────────────────────────────────
# WHY: Class Level (Freshman → Senior → Other) is the best available proxy
# for students' age and academic experience. Differences across levels reveal
# whether newer vs. more experienced students evaluate the programme differently.

print("\n=== C. Category scores by Class Level ===")
ORDER = ["Freshman", "Sophomore", "Junior", "Senior", "Other"]
cat_class = (
    likert_clean[likert_clean["class_level"].isin(ORDER)]
    .groupby(["class_level", "Section"])["score"]
    .mean()
    .reset_index()
    .rename(columns={"score": "mean_score"})
)

fig, axes = plt.subplots(1, len(CLEAN_SECTIONS), figsize=(18, 5), sharey=True)
for ax, sec in zip(axes, sorted(CLEAN_SECTIONS)):
    sub = (
        cat_class[cat_class["Section"] == sec]
        .set_index("class_level")
        .reindex(ORDER)
        .reset_index()
    )
    ax.bar(sub["class_level"], sub["mean_score"],
           color=sns.color_palette(PALETTE, len(ORDER)))
    ax.axhline(3, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(sec, fontsize=8, wrap=True)
    ax.set_ylim(1, 5)
    ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean Score" if ax == axes[0] else "")
fig.suptitle("Category Scores by Class Level (Age/Term Proxy)", fontsize=13)
# save_fig("C_category_by_class_level")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  LIKERT ANALYSIS D — Category scores by Term (Fall / Spring / Summer)
# ─────────────────────────────────────────────────────────────────────────────
# WHY: Students who study abroad in different terms may have systematically
# different experiences (e.g. Summer programmes are shorter; Fall may have
# richer activity calendars). This split surfaces those differences.

print("\n=== D. Category scores by Term ===")
cat_term = (
    likert_clean[likert_clean["term"] != "Unknown"]
    .groupby(["term", "Section"])["score"]
    .mean()
    .reset_index()
    .rename(columns={"score": "mean_score"})
)

fig, ax = plt.subplots(figsize=(10, 5))
pivot = cat_term.pivot(index="Section", columns="term", values="mean_score")
pivot.plot(kind="bar", ax=ax, color=sns.color_palette(PALETTE, 3), edgecolor="white")
ax.axhline(3, color="grey", linestyle="--", linewidth=1, label="Neutral")
ax.set_xlabel("")
ax.set_ylabel("Mean Score (1–5)")
ax.set_title("Category Scores by Term (Fall / Spring / Summer)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(title="Term")
# save_fig("D_category_by_term")

# ─────────────────────────────────────────────────────────────────────────────
# 10. LIKERT ANALYSIS E — Category scores by Cohort (Student Status)
# ─────────────────────────────────────────────────────────────────────────────
# WHY: There are two cohort types — regular study-abroad students and
# pre-freshman NUin students. These populations have very different
# expectations and contexts; comparing them shows whether programme
# elements serve both cohorts equally well.

print("\n=== E. Category scores by Cohort (Student Status) ===")
cat_cohort = (
    likert_clean.dropna(subset=["student_status"])
    .groupby(["student_status", "Section"])["score"]
    .mean()
    .reset_index()
    .rename(columns={"score": "mean_score"})
)

fig, ax = plt.subplots(figsize=(10, 5))
pivot_c = cat_cohort.pivot(index="Section", columns="student_status", values="mean_score")
pivot_c.plot(kind="bar", ax=ax, color=sns.color_palette(PALETTE, 2), edgecolor="white")
ax.axhline(3, color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("")
ax.set_ylabel("Mean Score (1–5)")
ax.set_title("Category Scores by Student Cohort")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(title="Cohort")
# save_fig("E_category_by_cohort")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  QUALITATIVE ANALYSIS — TF-IDF + NMF topic modelling
# ─────────────────────────────────────────────────────────────────────────────
"""
WHAT IS TF-IDF?
  Term Frequency–Inverse Document Frequency weights each word by how often
  it appears in a response (TF) divided by how common it is across ALL
  responses (IDF).  Common filler words ('the', 'a', 'it') get low IDF
  weight; distinctive words ('disappointing', 'helpful', 'overcrowded')
  get high weight.  We also use a custom stop-word list to remove survey-
  specific noise words that aren't meaningful on their own.

WHAT IS NMF?
  Non-negative Matrix Factorization decomposes the TF-IDF matrix into
  two non-negative matrices — one representing 'topics' as word mixtures,
  and one representing each response as a mixture of topics.  Because all
  values are ≥ 0, topics are additive and easily interpretable.

BIAS MITIGATION STEPS:
  1. No external sentiment lexicon — we do not import any "positive word"
     or "negative word" lists that were built on other populations.
  2. Custom stop words include survey boilerplate ('study', 'abroad',
     'program', 'experience') so the model finds content themes, not
     artefacts of the survey context.
  3. min_df=5 drops words that appear in fewer than 5 responses —
     idiosyncratic spellings and typos cannot form spurious topics.
  4. max_df=0.85 drops words that appear in >85 % of all responses —
     near-universal words add no discriminative power.
  5. Random state is fixed (seed=42) for full reproducibility.
  6. We normalise document-topic weights so that different response
     lengths don't give longer responses disproportionate influence.
"""

print("\n=== F. Qualitative NMF Topic Modelling ===")

# Drop empty / very short responses (< 10 chars carry no useful signal)
qual_clean = qual_df.dropna(subset=["ResponseText"]).copy()
qual_clean = qual_clean[qual_clean["ResponseText"].str.strip().str.len() >= 10]

print(f"Qualitative responses for modelling: {len(qual_clean):,}")

# Survey-specific stop words (not biased toward any demographic)
CUSTOM_STOP_WORDS = {
    "study", "abroad", "program", "programme", "experience", "student",
    "students", "act", "university", "college", "campus", "greece",
    "thessaloniki", "semester", "like", "really", "think", "also",
    "would", "feel", "felt", "lot", "much", "many", "time", "good",
    "great", "overall", "things", "thing", "way", "made", "make",
    "got", "get", "go", "went", "even", "one", "just", "want",
}

# ── Step 1: Build TF-IDF matrix ──────────────────────────────────────────────
# WHY these hyperparameters:
#   ngram_range=(1,2)  captures meaningful two-word phrases ('not helpful',
#                      'very friendly') without the combinatorial explosion
#                      of trigrams.
#   min_df=5           removes rare terms (likely typos/names).
#   max_df=0.85        removes terms so common they don't discriminate.
#   max_features=3000  keeps computation tractable; 3 k features is well
#                      above the vocabulary needed for survey text.

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

combined_stop_words = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS))

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.85,
    max_features=3000,
    stop_words=combined_stop_words,
)

tfidf_matrix = tfidf.fit_transform(qual_clean["ResponseText"])

# ── Step 2: Fit NMF ──────────────────────────────────────────────────────────
# n_components=8: eight topics is a practical sweet spot for a multi-section
# survey — enough granularity to distinguish categories without over-splitting.
# init='nndsvd' is deterministic (no random init) which further reduces
# run-to-run variance.

N_TOPICS = 8
nmf_model = NMF(
    n_components=N_TOPICS,
    init="nndsvd",    # deterministic initialisation
    random_state=42,
    max_iter=500,
)
doc_topics = nmf_model.fit_transform(tfidf_matrix)

# L1-normalise so each response has weights that sum to 1.
# This means a 3-sentence response and a 10-sentence response are
# on equal footing.
doc_topics_norm = normalize(doc_topics, norm="l1")

feature_names = tfidf.get_feature_names_out()

# ── Step 3: Label each topic by its top-8 keywords ───────────────────────────
def get_top_words(model, feature_names, n=8):
    topics = {}
    for i, comp in enumerate(model.components_):
        top_idx = comp.argsort()[-n:][::-1]
        topics[i] = [feature_names[j] for j in top_idx]
    return topics

topic_keywords = get_top_words(nmf_model, feature_names)
print("\nDiscovered topics (top keywords):")
for t, words in topic_keywords.items():
    print(f"  Topic {t}: {', '.join(words)}")

# Assign each response to its dominant topic
qual_clean = qual_clean.copy()
qual_clean["dominant_topic"] = doc_topics_norm.argmax(axis=1)
qual_clean["topic_label"] = qual_clean["dominant_topic"].map(
    {t: f"T{t}: {', '.join(kw[:3])}" for t, kw in topic_keywords.items()}
)

# ── Step 4: Visualise topic distribution ─────────────────────────────────────
topic_counts = qual_clean["topic_label"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    x=topic_counts.values, y=topic_counts.index,
    palette=PALETTE, ax=ax,
)
ax.set_xlabel("Number of Responses")
ax.set_title("Qualitative Response Topics (NMF, n=8)\nColour = topic index, labels show top 3 keywords")
# save_fig("F_qualitative_topics_overall")

# ── Step 5: Topic distribution by Year ───────────────────────────────────────
qual_year = (
    qual_clean.dropna(subset=["year"])
    .groupby(["year", "dominant_topic"])
    .size()
    .reset_index(name="count")
)
total_per_year = qual_year.groupby("year")["count"].transform("sum")
qual_year["pct"] = qual_year["count"] / total_per_year * 100

pivot_qy = qual_year.pivot(index="year", columns="dominant_topic", values="pct").fillna(0)
fig, ax = plt.subplots(figsize=(12, 6))
pivot_qy.plot(kind="bar", stacked=True, ax=ax,
              colormap="Set2", edgecolor="white")
ax.set_xlabel("Academic Year")
ax.set_ylabel("% of Responses")
ax.set_title("Qualitative Topic Mix by Year (% of open-ended responses)")
ax.legend(
    [f"T{t}: {', '.join(topic_keywords[t][:2])}" for t in pivot_qy.columns],
    loc="upper right", fontsize=8, title="Topic",
)
# save_fig("G_qualitative_topics_by_year")

# ── Step 6: Topic distribution by Class Level ─────────────────────────────────
qual_class = (
    qual_clean[qual_clean["class_level"].isin(ORDER)]
    .groupby(["class_level", "dominant_topic"])
    .size()
    .reset_index(name="count")
)
total_per_class = qual_class.groupby("class_level")["count"].transform("sum")
qual_class["pct"] = qual_class["count"] / total_per_class * 100

pivot_qc = (
    qual_class.pivot(index="class_level", columns="dominant_topic", values="pct")
    .fillna(0)
    .reindex(ORDER)
)
fig, ax = plt.subplots(figsize=(10, 5))
pivot_qc.plot(kind="bar", stacked=True, ax=ax,
              colormap="Set2", edgecolor="white")
ax.set_xlabel("Class Level")
ax.set_ylabel("% of Responses")
ax.set_title("Qualitative Topic Mix by Class Level")
ax.legend(
    [f"T{t}: {', '.join(topic_keywords[t][:2])}" for t in pivot_qc.columns],
    loc="upper right", fontsize=8, title="Topic",
)
# save_fig("H_qualitative_topics_by_class_level")

# ── Step 7: Topic distribution by Term ───────────────────────────────────────
qual_term = (
    qual_clean[qual_clean["term"] != "Unknown"]
    .groupby(["term", "dominant_topic"])
    .size()
    .reset_index(name="count")
)
total_per_term = qual_term.groupby("term")["count"].transform("sum")
qual_term["pct"] = qual_term["count"] / total_per_term * 100

pivot_qt = qual_term.pivot(index="term", columns="dominant_topic", values="pct").fillna(0)
fig, ax = plt.subplots(figsize=(9, 5))
pivot_qt.plot(kind="bar", stacked=True, ax=ax,
              colormap="Set2", edgecolor="white")
ax.set_xlabel("Term")
ax.set_ylabel("% of Responses")
ax.set_title("Qualitative Topic Mix by Term")
ax.legend(
    [f"T{t}: {', '.join(topic_keywords[t][:2])}" for t in pivot_qt.columns],
    loc="upper right", fontsize=8, title="Topic",
)
# save_fig("I_qualitative_topics_by_term")

# ── Step 8: Topic distribution by Cohort ─────────────────────────────────────
qual_cohort = (
    qual_clean.dropna(subset=["student_status"])
    .groupby(["student_status", "dominant_topic"])
    .size()
    .reset_index(name="count")
)
total_per_cohort = qual_cohort.groupby("student_status")["count"].transform("sum")
qual_cohort["pct"] = qual_cohort["count"] / total_per_cohort * 100

pivot_qcoh = qual_cohort.pivot(
    index="student_status", columns="dominant_topic", values="pct"
).fillna(0)
fig, ax = plt.subplots(figsize=(9, 5))
pivot_qcoh.plot(kind="bar", stacked=True, ax=ax,
                colormap="Set2", edgecolor="white")
ax.set_xlabel("Student Cohort")
ax.set_ylabel("% of Responses")
ax.set_title("Qualitative Topic Mix by Cohort (Student Status)")
ax.legend(
    [f"T{t}: {', '.join(topic_keywords[t][:2])}" for t in pivot_qcoh.columns],
    loc="upper right", fontsize=8, title="Topic",
)
# save_fig("J_qualitative_topics_by_cohort")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  SUMMARY TABLE — printed to console and saved as CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SUMMARY: Mean Likert Score per Section × Dimension ===")

def summary_pivot(groupby_col, clean_label):
    grp = (
        likert_clean.dropna(subset=[groupby_col])
        .groupby([groupby_col, "Section"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "mean_score", groupby_col: clean_label})
    )
    return grp.pivot(index="Section", columns=clean_label, values="mean_score").round(2)

print("\nBy Year:")
by_year = summary_pivot("year", "Year")
print(by_year)

print("\nBy Class Level:")
by_class = summary_pivot("class_level", "Class Level")
print(by_class)

print("\nBy Term:")
by_term = summary_pivot("term", "Term")
print(by_term)

print("\nBy Cohort:")
by_cohort = summary_pivot("student_status", "Cohort")
print(by_cohort)

# Save CSV summaries
for name, tbl in [
    ("summary_by_year", by_year),
    ("summary_by_class_level", by_class),
    ("summary_by_term", by_term),
    ("summary_by_cohort", by_cohort),
]:
    path = f"/mnt/user-data/outputs/{name}.csv"
    tbl.to_csv(path)
    print(f"  Saved CSV → {path}")

# Save topic keywords reference
kw_rows = [{"Topic": f"T{t}", "Keywords": ", ".join(words)}
           for t, words in topic_keywords.items()]
kw_df = pd.DataFrame(kw_rows)
kw_path = "/mnt/user-data/outputs/qualitative_topic_keywords.csv"
kw_df.to_csv(kw_path, index=False)
print(f"  Saved CSV → {kw_path}")

print("\n✓ All analysis complete.")