import pandas as pd
from pathlib import Path


# ============================================================
# STEP 13 - FINAL MODEL EVALUATION
# MULTIPLE LOGISTIC REGRESSION
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step7_model_parameters.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step8_test_predictions.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step10_evaluation_metrics.csv"
)

ROC_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step11_roc_auc.csv"
)

HISTORY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step7_training_history.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step13_final_model_summary.csv"
)


# ------------------------------------------------------------
# 2. LOAD MODEL PARAMETERS
# ------------------------------------------------------------

model_df = pd.read_csv(
    MODEL_FILE
)

model_parameters = dict(
    zip(
        model_df["parameter"],
        model_df["value"],
    )
)


b0 = model_parameters[
    "b0"
]

b1 = model_parameters[
    "b1_study_hours"
]

b2 = model_parameters[
    "b2_attendance"
]

b3 = model_parameters[
    "b3_previous_score"
]


# ------------------------------------------------------------
# 3. LOAD TEST PREDICTIONS
# ------------------------------------------------------------

predictions_df = pd.read_csv(
    PREDICTIONS_FILE
)

actual = predictions_df[
    "passed"
]

predicted = predictions_df[
    "predicted_class"
]


# ------------------------------------------------------------
# 4. CALCULATE CONFUSION MATRIX VALUES
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
# 5. LOAD CLASSIFICATION METRICS
# ------------------------------------------------------------

metrics_df = pd.read_csv(
    METRICS_FILE
)

metrics = dict(
    zip(
        metrics_df["metric"],
        metrics_df["value"],
    )
)


accuracy = metrics[
    "Accuracy"
]

precision = metrics[
    "Precision"
]

recall = metrics[
    "Recall"
]

f1_score = metrics[
    "F1_Score"
]


# ------------------------------------------------------------
# 6. LOAD ROC-AUC
# ------------------------------------------------------------

roc_df = pd.read_csv(
    ROC_FILE
)

roc_auc = roc_df.loc[
    roc_df["metric"] == "ROC_AUC",
    "value",
].iloc[0]


# ------------------------------------------------------------
# 7. LOAD TRAINING HISTORY
# ------------------------------------------------------------

history_df = pd.read_csv(
    HISTORY_FILE
)

initial_loss = history_df.iloc[
    0
]["loss"]

final_loss = history_df.iloc[
    -1
]["loss"]


# ------------------------------------------------------------
# 8. DISPLAY FINAL MODEL EQUATION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL MULTIPLE LOGISTIC REGRESSION MODEL")
print("=" * 80)

print(
    f"""
z =
{b0:.6f}
+ ({b1:.6f} × scaled_study_hours)
+ ({b2:.6f} × scaled_attendance)
+ ({b3:.6f} × scaled_previous_score)
"""
)

print(
    """
Then:

z
↓
Sigmoid
↓
Pass Probability
↓
Threshold = 0.5
↓
Fail or Pass
"""
)


# ------------------------------------------------------------
# 9. DISPLAY COEFFICIENTS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LEARNED COEFFICIENTS")
print("=" * 80)

print(
    f"Intercept       : {b0:.6f}"
)

print(
    f"Study Hours     : {b1:.6f}"
)

print(
    f"Attendance      : {b2:.6f}"
)

print(
    f"Previous Score  : {b3:.6f}"
)


# ------------------------------------------------------------
# 10. INTERPRET COEFFICIENT DIRECTIONS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("COEFFICIENT INTERPRETATION")
print("=" * 80)


coefficients = {
    "Study Hours": b1,
    "Attendance": b2,
    "Previous Score": b3,
}


for feature, coefficient in coefficients.items():

    if coefficient > 0:

        print(
            f"{feature}: "
            f"Higher values increase the model's "
            f"predicted probability of passing."
        )

    elif coefficient < 0:

        print(
            f"{feature}: "
            f"Higher values decrease the model's "
            f"predicted probability of passing."
        )

    else:

        print(
            f"{feature}: "
            f"No learned effect."
        )


# ------------------------------------------------------------
# 11. DISPLAY TRAINING LOSS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TRAINING LOSS")
print("=" * 80)

print(
    f"Initial Log Loss : {initial_loss:.6f}"
)

print(
    f"Final Log Loss   : {final_loss:.6f}"
)

print(
    f"Loss Reduction   : "
    f"{initial_loss - final_loss:.6f}"
)


# ------------------------------------------------------------
# 12. DISPLAY CONFUSION MATRIX
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(
    f"""
TP = {TP}
TN = {TN}
FP = {FP}
FN = {FN}
"""
)

print(
    f"""
                     PREDICTED

                  Fail      Pass

ACTUAL Fail        {TN}         {FP}

ACTUAL Pass        {FN}         {TP}
"""
)


# ------------------------------------------------------------
# 13. DISPLAY CLASSIFICATION METRICS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL CLASSIFICATION METRICS")
print("=" * 80)

print(
    f"Accuracy  : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision : {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall    : {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1 Score  : {f1_score:.4f} "
    f"({f1_score * 100:.2f}%)"
)

print(
    f"ROC-AUC   : {roc_auc:.4f} "
    f"({roc_auc * 100:.2f}%)"
)


# ------------------------------------------------------------
# 14. DISPLAY TEST DATA INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TEST DATA INFORMATION")
print("=" * 80)

print(
    f"Total Test Students : "
    f"{len(predictions_df)}"
)

print(
    f"Correct Predictions : "
    f"{TP + TN}"
)

print(
    f"Wrong Predictions   : "
    f"{FP + FN}"
)


# ------------------------------------------------------------
# 15. CREATE FINAL SUMMARY TABLE
# ------------------------------------------------------------

summary_df = pd.DataFrame(
    {
        "item": [
            "Intercept_b0",
            "Study_Hours_Coefficient_b1",
            "Attendance_Coefficient_b2",
            "Previous_Score_Coefficient_b3",
            "Initial_Log_Loss",
            "Final_Log_Loss",
            "Test_Samples",
            "True_Positive",
            "True_Negative",
            "False_Positive",
            "False_Negative",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "ROC_AUC",
        ],

        "value": [
            b0,
            b1,
            b2,
            b3,
            initial_loss,
            final_loss,
            len(predictions_df),
            TP,
            TN,
            FP,
            FN,
            accuracy,
            precision,
            recall,
            f1_score,
            roc_auc,
        ],
    }
)


# ------------------------------------------------------------
# 16. DISPLAY FINAL SUMMARY TABLE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL SUMMARY TABLE")
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 17. SAVE FINAL SUMMARY
# ------------------------------------------------------------

summary_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# 18. FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL INTERPRETATION")
print("=" * 80)

print(
    """
The Multiple Logistic Regression model used
three input features:

1. Study Hours
2. Attendance
3. Previous Score


The model learned one coefficient for each
feature using Gradient Descent.


All three learned coefficients were positive.

This means that, within this training dataset,
higher values of these features are associated
with a higher predicted probability of passing.


The model correctly classified all students
in the current test dataset.


However:

This is a small educational dataset.

Perfect test results do NOT mean that the
model would achieve perfect performance
on larger or real-world student data.
"""
)


# ------------------------------------------------------------
# 19. WHAT WAS NEW IN THIS PROJECT?
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHAT WAS NEW COMPARED WITH OUR FIRST PROJECT?")
print("=" * 80)

print(
    """
FIRST LOGISTIC REGRESSION PROJECT
---------------------------------

We used:

1 input feature

study_hours


The equation was:

z =
b0
+ b1 × study_hours



MULTIPLE LOGISTIC REGRESSION PROJECT
------------------------------------

We used:

3 input features

study_hours
attendance
previous_score


The equation became:

z =
b0
+ b1 × study_hours
+ b2 × attendance
+ b3 × previous_score


We also learned:

Feature Scaling
Training-only scaling parameters
Multiple coefficients
Multi-feature predictions
Training-loss visualization
"""
)


# ------------------------------------------------------------
# 20. DISPLAY SAVED FILE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL SUMMARY SAVED")
print("=" * 80)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 21. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 13 COMPLETED SUCCESSFULLY")
print("=" * 80)