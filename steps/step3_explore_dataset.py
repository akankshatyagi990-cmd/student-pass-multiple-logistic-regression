import pandas as pd
from pathlib import Path


# ============================================================
# STEP 3 - EXPLORE MULTIPLE LOGISTIC REGRESSION DATASET
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "student_pass_multiple_features.csv"
)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)


# ------------------------------------------------------------
# 3. DISPLAY FIRST 5 ROWS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FIRST 5 ROWS")
print("=" * 80)

print(
    df.head()
)


# ------------------------------------------------------------
# 4. DISPLAY LAST 5 ROWS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LAST 5 ROWS")
print("=" * 80)

print(
    df.tail()
)


# ------------------------------------------------------------
# 5. DISPLAY DATASET SHAPE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("DATASET SHAPE")
print("=" * 80)

print(
    f"Rows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)


# ------------------------------------------------------------
# 6. DISPLAY COLUMN NAMES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("COLUMN NAMES")
print("=" * 80)

for column in df.columns:
    print(f"- {column}")


# ------------------------------------------------------------
# 7. DISPLAY DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)

print(
    df.dtypes
)


# ------------------------------------------------------------
# 8. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

print(
    df.isnull().sum()
)


# ------------------------------------------------------------
# 9. CHECK DUPLICATE ROWS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("DUPLICATE ROWS")
print("=" * 80)

print(
    df.duplicated().sum()
)


# ------------------------------------------------------------
# 10. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

print(
    df.describe()
)


# ------------------------------------------------------------
# 11. CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)

class_counts = (
    df["passed"]
    .value_counts()
    .sort_index()
)

print(
    class_counts
)


# ------------------------------------------------------------
# 12. CLASS PERCENTAGES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CLASS PERCENTAGES")
print("=" * 80)

class_percentages = (
    df["passed"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print(
    class_percentages
)


# ------------------------------------------------------------
# 13. STUDY HOURS STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STUDY HOURS STATISTICS")
print("=" * 80)

print(
    f"Minimum : {df['study_hours'].min():.2f}"
)

print(
    f"Maximum : {df['study_hours'].max():.2f}"
)

print(
    f"Average : {df['study_hours'].mean():.2f}"
)


# ------------------------------------------------------------
# 14. ATTENDANCE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ATTENDANCE STATISTICS")
print("=" * 80)

print(
    f"Minimum : {df['attendance'].min():.2f}%"
)

print(
    f"Maximum : {df['attendance'].max():.2f}%"
)

print(
    f"Average : {df['attendance'].mean():.2f}%"
)


# ------------------------------------------------------------
# 15. PREVIOUS SCORE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PREVIOUS SCORE STATISTICS")
print("=" * 80)

print(
    f"Minimum : {df['previous_score'].min():.2f}"
)

print(
    f"Maximum : {df['previous_score'].max():.2f}"
)

print(
    f"Average : {df['previous_score'].mean():.2f}"
)


# ------------------------------------------------------------
# 16. AVERAGE FEATURES BY CLASS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("AVERAGE FEATURES BY PASS / FAIL")
print("=" * 80)

average_by_class = (
    df.groupby("passed")[
        [
            "study_hours",
            "attendance",
            "previous_score",
        ]
    ]
    .mean()
)

print(
    average_by_class
)


# ------------------------------------------------------------
# 17. CORRELATION WITH TARGET
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CORRELATION WITH PASSED")
print("=" * 80)

correlations = (
    df[
        [
            "study_hours",
            "attendance",
            "previous_score",
            "passed",
        ]
    ]
    .corr()["passed"]
    .sort_values(ascending=False)
)

print(
    correlations
)


# ------------------------------------------------------------
# 18. DISPLAY FIRST 10 RELEVANT ROWS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FIRST 10 RELEVANT ROWS")
print("=" * 80)

print(
    df[
        [
            "study_hours",
            "attendance",
            "previous_score",
            "passed",
        ]
    ].head(10)
)


# ------------------------------------------------------------
# 19. EXPLAIN WHAT TO OBSERVE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHAT TO OBSERVE")
print("=" * 80)

print(
    """
Look at the average values for students who:

0 = Failed
1 = Passed

Ask yourself:

Do students who pass generally have:

- More study hours?
- Higher attendance?
- Higher previous scores?

Also look at the correlation values.

A positive correlation with 'passed'
means that as the feature increases,
passing tends to become more likely.
"""
)


# ------------------------------------------------------------
# 20. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 80)