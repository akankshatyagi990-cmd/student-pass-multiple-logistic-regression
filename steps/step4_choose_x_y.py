import pandas as pd
from pathlib import Path


# ============================================================
# STEP 4 - CHOOSE X AND y
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
# 3. SELECT INPUT FEATURES
# ------------------------------------------------------------

X = df[
    [
        "study_hours",
        "attendance",
        "previous_score",
    ]
]


# ------------------------------------------------------------
# 4. SELECT TARGET
# ------------------------------------------------------------

y = df["passed"]


# ------------------------------------------------------------
# 5. DISPLAY X
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("INPUT FEATURES - X")
print("=" * 80)

print(
    X.head(10)
)


# ------------------------------------------------------------
# 6. DISPLAY y
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TARGET - y")
print("=" * 80)

print(
    y.head(10)
)


# ------------------------------------------------------------
# 7. DISPLAY X SHAPE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("X SHAPE")
print("=" * 80)

print(
    f"Rows     : {X.shape[0]}"
)

print(
    f"Features : {X.shape[1]}"
)


# ------------------------------------------------------------
# 8. DISPLAY y SHAPE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("y SHAPE")
print("=" * 80)

print(
    f"Rows : {y.shape[0]}"
)


# ------------------------------------------------------------
# 9. DISPLAY FEATURE NAMES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FEATURE NAMES")
print("=" * 80)

for feature in X.columns:
    print(
        f"- {feature}"
    )


# ------------------------------------------------------------
# 10. DISPLAY TARGET NAME
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TARGET NAME")
print("=" * 80)

print(
    y.name
)


# ------------------------------------------------------------
# 11. EXPLAIN THE MODEL INPUT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHAT DID WE DO?")
print("=" * 80)

print(
    """
We separated the dataset into:

X = Input Features

X1 = study_hours
X2 = attendance
X3 = previous_score


y = Target

y = passed


The model will use the three values in X
to predict whether the student:

0 = Fails
1 = Passes
"""
)


# ------------------------------------------------------------
# 12. EXPLAIN THE EQUATION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("MULTIPLE LOGISTIC REGRESSION EQUATION")
print("=" * 80)

print(
    """
Our model will eventually learn:

z =
b0
+ b1 × study_hours
+ b2 × attendance
+ b3 × previous_score


Then:

z
↓
Sigmoid
↓
Probability
↓
Threshold
↓
0 or 1
"""
)


# ------------------------------------------------------------
# 13. EXPLAIN PARAMETERS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PARAMETERS THE MODEL WILL LEARN")
print("=" * 80)

print(
    """
b0 = Intercept

b1 = Coefficient for study_hours

b2 = Coefficient for attendance

b3 = Coefficient for previous_score
"""
)


# ------------------------------------------------------------
# 14. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 80)