import pandas as pd
from pathlib import Path


# ============================================================
# STEP 2 - CREATE MULTIPLE LOGISTIC REGRESSION DATASET
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "student_pass_multiple_features.csv"
)


# ------------------------------------------------------------
# 2. CREATE SAMPLE STUDENT DATA
# ------------------------------------------------------------

data = {
    "student_id": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25,
        26, 27, 28, 29, 30,
        31, 32, 33, 34, 35,
        36, 37, 38, 39, 40,
    ],

    "study_hours": [
        1.0, 1.2, 1.5, 1.7, 2.0,
        2.1, 2.3, 2.5, 2.8, 3.0,
        3.2, 3.4, 3.5, 3.7, 3.9,
        4.0, 4.2, 4.4, 4.5, 4.7,
        4.8, 5.0, 5.2, 5.5, 5.7,
        5.8, 6.0, 6.2, 6.5, 6.8,
        7.0, 7.2, 7.5, 7.8, 8.0,
        8.2, 8.5, 8.8, 9.0, 9.5,
    ],

    "attendance": [
        52, 55, 58, 60, 62,
        65, 67, 70, 72, 74,
        76, 78, 80, 68, 82,
        84, 75, 86, 88, 79,
        90, 83, 85, 91, 87,
        92, 89, 93, 94, 90,
        95, 91, 96, 94, 97,
        95, 98, 97, 99, 99,
    ],

    "previous_score": [
        38, 40, 42, 44, 46,
        48, 50, 52, 54, 56,
        58, 60, 62, 55, 64,
        66, 59, 68, 70, 63,
        72, 67, 69, 74, 71,
        76, 73, 78, 80, 75,
        82, 77, 84, 81, 86,
        83, 88, 85, 90, 92,
    ],

    "passed": [
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 1,
        1, 0, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
    ],
}


# ------------------------------------------------------------
# 3. CONVERT TO DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame(data)


# ------------------------------------------------------------
# 4. DISPLAY DATASET
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("MULTIPLE LOGISTIC REGRESSION DATASET")
print("=" * 80)

print(df)


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
# 6. DISPLAY FEATURE MEANING
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FEATURES")
print("=" * 80)

print(
    """
X1 = study_hours
X2 = attendance
X3 = previous_score

Target:

y = passed

0 = Fail
1 = Pass
"""
)


# ------------------------------------------------------------
# 7. DISPLAY CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)

print(
    df["passed"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 8. DISPLAY AVERAGES BY CLASS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("AVERAGE FEATURE VALUES BY CLASS")
print("=" * 80)

print(
    df.groupby("passed")[
        [
            "study_hours",
            "attendance",
            "previous_score",
        ]
    ].mean()
)


# ------------------------------------------------------------
# 9. SAVE DATASET
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# 10. DISPLAY SAVED LOCATION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("DATASET SAVED")
print("=" * 80)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 11. EXPLAIN THE MODEL
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("OUR NEW LOGISTIC REGRESSION MODEL")
print("=" * 80)

print(
    """
Previously we used only:

study_hours


Now we will use:

study_hours
attendance
previous_score


The model will eventually learn:

z =
b0
+ b1 × study_hours
+ b2 × attendance
+ b3 × previous_score

Then:

z
→ sigmoid
→ probability
→ threshold
→ Fail or Pass
"""
)


# ------------------------------------------------------------
# 12. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 80)