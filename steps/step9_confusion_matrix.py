import pandas as pd
from pathlib import Path


# ============================================================
# STEP 9 - CONFUSION MATRIX
# MULTIPLE LOGISTIC REGRESSION
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step8_test_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step9_confusion_matrix.csv"
)


# ------------------------------------------------------------
# 2. LOAD TEST PREDICTIONS
# ------------------------------------------------------------

df = pd.read_csv(
    PREDICTIONS_FILE
)


# ------------------------------------------------------------
# 3. GET ACTUAL AND PREDICTED CLASSES
# ------------------------------------------------------------

actual = df[
    "passed"
]

predicted = df[
    "predicted_class"
]


# ------------------------------------------------------------
# 4. CALCULATE TRUE POSITIVES
# ------------------------------------------------------------

TP = (
    (actual == 1)
    &
    (predicted == 1)
).sum()


# ------------------------------------------------------------
# 5. CALCULATE TRUE NEGATIVES
# ------------------------------------------------------------

TN = (
    (actual == 0)
    &
    (predicted == 0)
).sum()


# ------------------------------------------------------------
# 6. CALCULATE FALSE POSITIVES
# ------------------------------------------------------------

FP = (
    (actual == 0)
    &
    (predicted == 1)
).sum()


# ------------------------------------------------------------
# 7. CALCULATE FALSE NEGATIVES
# ------------------------------------------------------------

FN = (
    (actual == 1)
    &
    (predicted == 0)
).sum()


# ------------------------------------------------------------
# 8. DISPLAY VALUES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CONFUSION MATRIX VALUES")
print("=" * 80)

print(
    f"TP = {TP}"
)

print(
    f"TN = {TN}"
)

print(
    f"FP = {FP}"
)

print(
    f"FN = {FN}"
)


# ------------------------------------------------------------
# 9. DISPLAY CONFUSION MATRIX
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(
    f"""
                     PREDICTED

                  Fail      Pass

ACTUAL Fail        {TN}         {FP}

ACTUAL Pass        {FN}         {TP}
"""
)


# ------------------------------------------------------------
# 10. CALCULATE CORRECT AND WRONG PREDICTIONS
# ------------------------------------------------------------

correct_predictions = (
    TP + TN
)

wrong_predictions = (
    FP + FN
)

total_predictions = (
    TP + TN + FP + FN
)


# ------------------------------------------------------------
# 11. DISPLAY SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Correct Predictions = {correct_predictions}"
)

print(
    f"Wrong Predictions   = {wrong_predictions}"
)

print(
    f"Total Predictions   = {total_predictions}"
)


# ------------------------------------------------------------
# 12. EXPLAIN EACH TERM
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHAT DO TP, TN, FP AND FN MEAN?")
print("=" * 80)

print(
    f"""
TP = True Positive = {TP}

Actual result    = Pass
Predicted result = Pass

The model correctly predicted a student
would pass.


TN = True Negative = {TN}

Actual result    = Fail
Predicted result = Fail

The model correctly predicted a student
would fail.


FP = False Positive = {FP}

Actual result    = Fail
Predicted result = Pass

The model incorrectly predicted Pass.


FN = False Negative = {FN}

Actual result    = Pass
Predicted result = Fail

The model incorrectly predicted Fail.
"""
)


# ------------------------------------------------------------
# 13. CREATE CONFUSION MATRIX TABLE
# ------------------------------------------------------------

confusion_matrix_df = pd.DataFrame(
    {
        "Actual_Class": [
            "Fail",
            "Fail",
            "Pass",
            "Pass",
        ],

        "Predicted_Class": [
            "Fail",
            "Pass",
            "Fail",
            "Pass",
        ],

        "Type": [
            "True Negative",
            "False Positive",
            "False Negative",
            "True Positive",
        ],

        "Count": [
            TN,
            FP,
            FN,
            TP,
        ],
    }
)


# ------------------------------------------------------------
# 14. DISPLAY TABLE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CONFUSION MATRIX TABLE")
print("=" * 80)

print(
    confusion_matrix_df
)


# ------------------------------------------------------------
# 15. SAVE CONFUSION MATRIX
# ------------------------------------------------------------

confusion_matrix_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# 16. EXPLAIN WHY THIS IS IMPORTANT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHY IS THE CONFUSION MATRIX IMPORTANT?")
print("=" * 80)

print(
    """
Only knowing how many predictions are correct
is not always enough.

The Confusion Matrix tells us WHICH TYPE
of mistake the model made.

For example:

False Positive
and
False Negative

can have very different importance depending
on the real-world problem.

We will use TP, TN, FP and FN in the next step
to calculate:

Accuracy
Precision
Recall
F1 Score
"""
)


# ------------------------------------------------------------
# 17. DISPLAY SAVED FILE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FILE SAVED")
print("=" * 80)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 18. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 9 COMPLETED SUCCESSFULLY")
print("=" * 80)