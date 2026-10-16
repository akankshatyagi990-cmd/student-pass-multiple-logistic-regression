import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 5 - TRAIN / TEST SPLIT
# MULTIPLE LOGISTIC REGRESSION
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "student_pass_multiple_features.csv"
)

TRAIN_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step5_train_data.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step5_test_data.csv"
)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)


# ------------------------------------------------------------
# 3. SEPARATE FAIL AND PASS CLASSES
# ------------------------------------------------------------

fail_students = df[
    df["passed"] == 0
].copy()

pass_students = df[
    df["passed"] == 1
].copy()


# ------------------------------------------------------------
# 4. SHUFFLE EACH CLASS
# ------------------------------------------------------------

np.random.seed(42)

fail_students = (
    fail_students
    .sample(
        frac=1,
        random_state=42,
    )
    .reset_index(drop=True)
)

pass_students = (
    pass_students
    .sample(
        frac=1,
        random_state=42,
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 5. CALCULATE 80% TRAINING SIZE FOR EACH CLASS
# ------------------------------------------------------------

fail_train_size = int(
    len(fail_students) * 0.80
)

pass_train_size = int(
    len(pass_students) * 0.80
)


# ------------------------------------------------------------
# 6. CREATE TRAINING DATA
# ------------------------------------------------------------

fail_train = fail_students.iloc[
    :fail_train_size
]

pass_train = pass_students.iloc[
    :pass_train_size
]


train_df = pd.concat(
    [
        fail_train,
        pass_train,
    ],
    ignore_index=True,
)


# ------------------------------------------------------------
# 7. CREATE TEST DATA
# ------------------------------------------------------------

fail_test = fail_students.iloc[
    fail_train_size:
]

pass_test = pass_students.iloc[
    pass_train_size:
]


test_df = pd.concat(
    [
        fail_test,
        pass_test,
    ],
    ignore_index=True,
)


# ------------------------------------------------------------
# 8. SHUFFLE FINAL TRAINING DATA
# ------------------------------------------------------------

train_df = (
    train_df
    .sample(
        frac=1,
        random_state=42,
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 9. SHUFFLE FINAL TEST DATA
# ------------------------------------------------------------

test_df = (
    test_df
    .sample(
        frac=1,
        random_state=42,
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 10. DISPLAY ORIGINAL DATA INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ORIGINAL DATASET")
print("=" * 80)

print(
    f"Total Students : {len(df)}"
)

print("\nClass Distribution:")

print(
    df["passed"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 11. DISPLAY TRAINING DATA INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TRAINING DATA")
print("=" * 80)

print(
    f"Training Students : {len(train_df)}"
)

print("\nTraining Class Distribution:")

print(
    train_df["passed"]
    .value_counts()
    .sort_index()
)


print("\nFirst 10 Training Rows:")

print(
    train_df.head(10)
)


# ------------------------------------------------------------
# 12. DISPLAY TEST DATA INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TEST DATA")
print("=" * 80)

print(
    f"Test Students : {len(test_df)}"
)

print("\nTest Class Distribution:")

print(
    test_df["passed"]
    .value_counts()
    .sort_index()
)


print("\nTest Data:")

print(
    test_df
)


# ------------------------------------------------------------
# 13. CHECK TOTAL ROWS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SPLIT CHECK")
print("=" * 80)

print(
    f"Original Rows       : {len(df)}"
)

print(
    f"Training Rows       : {len(train_df)}"
)

print(
    f"Test Rows           : {len(test_df)}"
)

print(
    f"Train + Test Rows   : {len(train_df) + len(test_df)}"
)


# ------------------------------------------------------------
# 14. SAVE TRAINING DATA
# ------------------------------------------------------------

train_df.to_csv(
    TRAIN_FILE,
    index=False,
)


# ------------------------------------------------------------
# 15. SAVE TEST DATA
# ------------------------------------------------------------

test_df.to_csv(
    TEST_FILE,
    index=False,
)


# ------------------------------------------------------------
# 16. EXPLAIN WHAT WE DID
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHAT DID WE DO?")
print("=" * 80)

print(
    """
We divided the complete dataset into:

TRAINING DATA
-------------
Used by the Logistic Regression model
to learn its coefficients.

b0
b1
b2
b3


TEST DATA
---------
Kept separate from training.

The model will later make predictions
on these unseen students so that we can
evaluate its performance.


We also kept both classes:

0 = Fail
1 = Pass

inside both the training and test datasets.
"""
)


# ------------------------------------------------------------
# 17. IMPORTANT NOTE FOR NEXT STEP
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHY DID WE SPLIT BEFORE FEATURE SCALING?")
print("=" * 80)

print(
    """
Our three features have very different ranges.

For example:

study_hours
≈ 1 to 10

attendance
≈ 50 to 100

previous_score
≈ 40 to 90


Before training, we will scale these features.

But we must learn the scaling information
ONLY from the training data.

We should not use information from the
test data while preparing the model.

Otherwise, we can create:

DATA LEAKAGE
"""
)


# ------------------------------------------------------------
# 18. DISPLAY SAVED FILES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Training Data : {TRAIN_FILE}"
)

print(
    f"Test Data     : {TEST_FILE}"
)


# ------------------------------------------------------------
# 19. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 80)