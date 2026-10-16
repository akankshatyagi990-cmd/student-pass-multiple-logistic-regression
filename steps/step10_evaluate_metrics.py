import pandas as pd
from pathlib import Path


# ============================================================
# STEP 10 - EVALUATE CLASSIFICATION METRICS
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
    / "step10_evaluation_metrics.csv"
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
# 4. CALCULATE TP, TN, FP, FN
# ------------------------------------------------------------

TP = (
    (actual == 1)
    &
    (predicted == 1)
).sum()


TN = (
    (actual == 0)
    &
    (predicted == 0)
).sum()


FP = (
    (actual == 0)
    &
    (predicted == 1)
).sum()


FN = (
    (actual == 1)
    &
    (predicted == 0)
).sum()


# ------------------------------------------------------------
# 5. CALCULATE TOTAL PREDICTIONS
# ------------------------------------------------------------

total = (
    TP
    + TN
    + FP
    + FN
)


# ------------------------------------------------------------
# 6. CALCULATE ACCURACY
# ------------------------------------------------------------

accuracy = (
    (TP + TN)
    / total
)


# ------------------------------------------------------------
# 7. CALCULATE PRECISION
# ------------------------------------------------------------

if (TP + FP) == 0:

    precision = 0.0

else:

    precision = (
        TP
        / (TP + FP)
    )


# ------------------------------------------------------------
# 8. CALCULATE RECALL
# ------------------------------------------------------------

if (TP + FN) == 0:

    recall = 0.0

else:

    recall = (
        TP
        / (TP + FN)
    )


# ------------------------------------------------------------
# 9. CALCULATE F1 SCORE
# ------------------------------------------------------------

if (precision + recall) == 0:

    f1_score = 0.0

else:

    f1_score = (
        2
        * precision
        * recall
        / (precision + recall)
    )


# ------------------------------------------------------------
# 10. DISPLAY CONFUSION MATRIX VALUES
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
# 11. DISPLAY ACCURACY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ACCURACY")
print("=" * 80)

print(
    f"Accuracy = {accuracy:.4f}"
)

print(
    f"Accuracy = {accuracy * 100:.2f}%"
)

print(
    """
Formula:

Accuracy
=
(TP + TN)
/
(TP + TN + FP + FN)
"""
)


# ------------------------------------------------------------
# 12. DISPLAY PRECISION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PRECISION")
print("=" * 80)

print(
    f"Precision = {precision:.4f}"
)

print(
    f"Precision = {precision * 100:.2f}%"
)

print(
    """
Formula:

Precision
=
TP
/
(TP + FP)


Meaning:

Out of all students predicted as Pass,
how many actually passed?
"""
)


# ------------------------------------------------------------
# 13. DISPLAY RECALL
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("RECALL")
print("=" * 80)

print(
    f"Recall = {recall:.4f}"
)

print(
    f"Recall = {recall * 100:.2f}%"
)

print(
    """
Formula:

Recall
=
TP
/
(TP + FN)


Meaning:

Out of all students who actually passed,
how many did the model correctly identify?
"""
)


# ------------------------------------------------------------
# 14. DISPLAY F1 SCORE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("F1 SCORE")
print("=" * 80)

print(
    f"F1 Score = {f1_score:.4f}"
)

print(
    f"F1 Score = {f1_score * 100:.2f}%"
)

print(
    """
F1 Score combines:

Precision
and
Recall


It is useful when we want a balance
between both metrics.
"""
)


# ------------------------------------------------------------
# 15. CREATE FINAL METRICS TABLE
# ------------------------------------------------------------

metrics_df = pd.DataFrame(
    {
        "metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
        ],

        "value": [
            accuracy,
            precision,
            recall,
            f1_score,
        ],
    }
)


# ------------------------------------------------------------
# 16. DISPLAY FINAL TABLE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL METRICS TABLE")
print("=" * 80)

print(
    metrics_df
)


# ------------------------------------------------------------
# 17. SAVE METRICS
# ------------------------------------------------------------

metrics_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# 18. INTERPRET RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(
    f"""
Accuracy  = {accuracy * 100:.2f}%
Precision = {precision * 100:.2f}%
Recall    = {recall * 100:.2f}%
F1 Score  = {f1_score * 100:.2f}%


Remember:

This is a small educational dataset.

Perfect scores on this small test set
do not mean the model will always achieve
perfect performance on real-world data.
"""
)


# ------------------------------------------------------------
# 19. DISPLAY SAVED FILE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FILE SAVED")
print("=" * 80)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 20. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 10 COMPLETED SUCCESSFULLY")
print("=" * 80)