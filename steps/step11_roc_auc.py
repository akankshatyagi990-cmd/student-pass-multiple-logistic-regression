import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 11 - ROC CURVE AND ROC-AUC
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

ROC_POINTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step11_roc_points.csv"
)

ROC_AUC_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step11_roc_auc.csv"
)

PLOT_FILE = (
    PROJECT_ROOT
    / "plots"
    / "step11_roc_curve.png"
)


# ------------------------------------------------------------
# 2. LOAD TEST PREDICTIONS
# ------------------------------------------------------------

df = pd.read_csv(
    PREDICTIONS_FILE
)


# ------------------------------------------------------------
# 3. GET ACTUAL CLASSES AND PROBABILITIES
# ------------------------------------------------------------

actual = (
    df["passed"]
    .to_numpy()
)

probabilities = (
    df["pass_probability"]
    .to_numpy()
)


# ------------------------------------------------------------
# 4. DISPLAY TEST PROBABILITIES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TEST PROBABILITIES")
print("=" * 80)

probability_table = pd.DataFrame(
    {
        "actual_class": actual,
        "pass_probability": probabilities,
    }
)

print(
    probability_table
)


# ------------------------------------------------------------
# 5. CREATE THRESHOLDS
# ------------------------------------------------------------

# Start above the maximum possible probability.
# End below the minimum possible probability.

thresholds = np.concatenate(
    (
        [1.01],
        np.sort(probabilities)[::-1],
        [-0.01],
    )
)


# ------------------------------------------------------------
# 6. PREPARE ROC POINTS
# ------------------------------------------------------------

roc_points = []


# ------------------------------------------------------------
# 7. TEST EVERY THRESHOLD
# ------------------------------------------------------------

for threshold in thresholds:

    predicted = (
        probabilities
        >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # CONFUSION MATRIX VALUES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # TRUE POSITIVE RATE
    # --------------------------------------------------------

    if (TP + FN) == 0:

        TPR = 0.0

    else:

        TPR = (
            TP
            / (TP + FN)
        )


    # --------------------------------------------------------
    # FALSE POSITIVE RATE
    # --------------------------------------------------------

    if (FP + TN) == 0:

        FPR = 0.0

    else:

        FPR = (
            FP
            / (FP + TN)
        )


    # --------------------------------------------------------
    # SAVE ROC POINT
    # --------------------------------------------------------

    roc_points.append(
        {
            "threshold": threshold,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "FPR": FPR,
            "TPR": TPR,
        }
    )


# ------------------------------------------------------------
# 8. CREATE ROC DATAFRAME
# ------------------------------------------------------------

roc_df = pd.DataFrame(
    roc_points
)


# ------------------------------------------------------------
# 9. REMOVE DUPLICATE ROC POINTS
# ------------------------------------------------------------

roc_df = (
    roc_df
    .drop_duplicates(
        subset=[
            "FPR",
            "TPR",
        ]
    )
    .sort_values(
        by=[
            "FPR",
            "TPR",
        ]
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 10. DISPLAY ROC POINTS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ROC POINTS")
print("=" * 80)

print(
    roc_df[
        [
            "threshold",
            "FPR",
            "TPR",
        ]
    ]
)


# ------------------------------------------------------------
# 11. CALCULATE ROC-AUC
# ------------------------------------------------------------

fpr_values = (
    roc_df["FPR"]
    .to_numpy()
)

tpr_values = (
    roc_df["TPR"]
    .to_numpy()
)


roc_auc = np.trapezoid(
    tpr_values,
    fpr_values,
)


# ------------------------------------------------------------
# 12. DISPLAY ROC-AUC
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ROC-AUC")
print("=" * 80)

print(
    f"ROC-AUC = {roc_auc:.4f}"
)

print(
    f"ROC-AUC = {roc_auc * 100:.2f}%"
)


# ------------------------------------------------------------
# 13. INTERPRET ROC-AUC
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)


if roc_auc >= 0.90:

    print(
        "The model has excellent class-separation ability."
    )

elif roc_auc >= 0.80:

    print(
        "The model has good class-separation ability."
    )

elif roc_auc >= 0.70:

    print(
        "The model has acceptable class-separation ability."
    )

elif roc_auc >= 0.60:

    print(
        "The model has weak class-separation ability."
    )

else:

    print(
        "The model performs close to random guessing."
    )


print(
    """
Remember:

ROC-AUC near 1.0
→ Excellent separation between Pass and Fail.

ROC-AUC near 0.5
→ Similar to random guessing.
"""
)


# ------------------------------------------------------------
# 14. CREATE ROC CURVE
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)


plt.plot(
    roc_df["FPR"],
    roc_df["TPR"],
    marker="o",
    label=f"Multiple Logistic Regression (AUC = {roc_auc:.4f})",
)


# Random guessing line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Guessing",
)


plt.xlabel(
    "False Positive Rate (FPR)"
)

plt.ylabel(
    "True Positive Rate (TPR)"
)

plt.title(
    "ROC Curve - Multiple Logistic Regression"
)

plt.xlim(
    0,
    1,
)

plt.ylim(
    0,
    1.05,
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()


# ------------------------------------------------------------
# 15. SAVE ROC CURVE
# ------------------------------------------------------------

plt.savefig(
    PLOT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# 16. SAVE ROC POINTS
# ------------------------------------------------------------

roc_df.to_csv(
    ROC_POINTS_FILE,
    index=False,
)


# ------------------------------------------------------------
# 17. SAVE ROC-AUC VALUE
# ------------------------------------------------------------

roc_auc_df = pd.DataFrame(
    {
        "metric": [
            "ROC_AUC"
        ],

        "value": [
            roc_auc
        ],
    }
)


roc_auc_df.to_csv(
    ROC_AUC_FILE,
    index=False,
)


# ------------------------------------------------------------
# 18. EXPLAIN WHAT WE LEARNED
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHAT DID WE LEARN?")
print("=" * 80)

print(
    """
Accuracy, Precision, Recall and F1 Score
used our selected threshold of 0.5.

ROC analysis is different.

It changes the classification threshold
and checks how:

True Positive Rate

and

False Positive Rate

change.


ROC-AUC summarizes the model's ability
to rank Pass students above Fail students
across different thresholds.
"""
)


# ------------------------------------------------------------
# 19. IMPORTANT NOTE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    """
Our test dataset contains only 9 students.

Therefore, even if ROC-AUC is very high,
we should not conclude that the model will
perform perfectly on larger real-world data.
"""
)


# ------------------------------------------------------------
# 20. DISPLAY SAVED FILES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"ROC Points : {ROC_POINTS_FILE}"
)

print(
    f"ROC-AUC    : {ROC_AUC_FILE}"
)

print(
    f"ROC Plot   : {PLOT_FILE}"
)


# ------------------------------------------------------------
# 21. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 11 COMPLETED SUCCESSFULLY")
print("=" * 80)