import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 8 - PREDICT TEST DATA
# MULTIPLE LOGISTIC REGRESSION
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step6_scaled_test_data.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step7_model_parameters.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step8_test_predictions.csv"
)


# ------------------------------------------------------------
# 2. LOAD SCALED TEST DATA
# ------------------------------------------------------------

test_df = pd.read_csv(TEST_FILE)


# ------------------------------------------------------------
# 3. LOAD MODEL PARAMETERS
# ------------------------------------------------------------

model_df = pd.read_csv(MODEL_FILE)


model_parameters = dict(
    zip(
        model_df["parameter"],
        model_df["value"],
    )
)


# ------------------------------------------------------------
# 4. GET LEARNED PARAMETERS
# ------------------------------------------------------------

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
# 5. DISPLAY MODEL PARAMETERS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LOADED MODEL PARAMETERS")
print("=" * 80)

print(
    f"b0 = {b0:.6f}"
)

print(
    f"b1 = {b1:.6f}  (study_hours)"
)

print(
    f"b2 = {b2:.6f}  (attendance)"
)

print(
    f"b3 = {b3:.6f}  (previous_score)"
)


# ------------------------------------------------------------
# 6. CREATE SIGMOID FUNCTION
# ------------------------------------------------------------

def sigmoid(z):

    z = np.clip(
        z,
        -500,
        500,
    )

    return 1 / (
        1 + np.exp(-z)
    )


# ------------------------------------------------------------
# 7. CALCULATE z FOR EVERY TEST STUDENT
# ------------------------------------------------------------

z = (
    b0
    + b1 * test_df["study_hours"]
    + b2 * test_df["attendance"]
    + b3 * test_df["previous_score"]
)


# ------------------------------------------------------------
# 8. CALCULATE PASS PROBABILITY
# ------------------------------------------------------------

probabilities = sigmoid(
    z
)


# ------------------------------------------------------------
# 9. APPLY DECISION THRESHOLD
# ------------------------------------------------------------

threshold = 0.5


predicted_classes = (
    probabilities
    >= threshold
).astype(int)


# ------------------------------------------------------------
# 10. CREATE RESULT TABLE
# ------------------------------------------------------------

results = test_df.copy()


results["z"] = z

results["pass_probability"] = probabilities

results["predicted_class"] = predicted_classes


# ------------------------------------------------------------
# 11. ADD ACTUAL AND PREDICTED LABELS
# ------------------------------------------------------------

results["actual_result"] = (
    results["passed"]
    .map(
        {
            0: "Fail",
            1: "Pass",
        }
    )
)


results["predicted_result"] = (
    results["predicted_class"]
    .map(
        {
            0: "Fail",
            1: "Pass",
        }
    )
)


# ------------------------------------------------------------
# 12. CHECK WHETHER PREDICTION IS CORRECT
# ------------------------------------------------------------

results["correct_prediction"] = (
    results["passed"]
    == results["predicted_class"]
)


# ------------------------------------------------------------
# 13. DISPLAY TEST PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TEST PREDICTIONS")
print("=" * 80)


display_columns = [
    "student_id",
    "study_hours",
    "attendance",
    "previous_score",
    "passed",
    "z",
    "pass_probability",
    "predicted_class",
    "actual_result",
    "predicted_result",
    "correct_prediction",
]


print(
    results[
        display_columns
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 14. DISPLAY SIMPLE PREDICTION SUMMARY
# ------------------------------------------------------------

correct_predictions = (
    results["correct_prediction"]
    .sum()
)

wrong_predictions = (
    len(results)
    - correct_predictions
)


print("\n" + "=" * 80)
print("PREDICTION SUMMARY")
print("=" * 80)

print(
    f"Total Test Students : {len(results)}"
)

print(
    f"Correct Predictions : {correct_predictions}"
)

print(
    f"Wrong Predictions   : {wrong_predictions}"
)


# ------------------------------------------------------------
# 15. EXPLAIN ONE STUDENT
# ------------------------------------------------------------

example = results.iloc[0]


print("\n" + "=" * 80)
print("ONE STUDENT EXAMPLE")
print("=" * 80)

print(
    f"Student ID             : "
    f"{int(example['student_id'])}"
)

print(
    f"Scaled Study Hours     : "
    f"{example['study_hours']:.6f}"
)

print(
    f"Scaled Attendance      : "
    f"{example['attendance']:.6f}"
)

print(
    f"Scaled Previous Score  : "
    f"{example['previous_score']:.6f}"
)

print(
    f"z                      : "
    f"{example['z']:.6f}"
)

print(
    f"Pass Probability       : "
    f"{example['pass_probability']:.6f}"
)

print(
    f"Predicted Class        : "
    f"{int(example['predicted_class'])}"
)

print(
    f"Actual Class           : "
    f"{int(example['passed'])}"
)


# ------------------------------------------------------------
# 16. EXPLAIN HOW THE PREDICTION WAS MADE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("HOW WAS EACH PREDICTION MADE?")
print("=" * 80)

print(
    """
For every unseen test student:

1. Take the three SCALED feature values.

2. Calculate:

   z =
   b0
   + b1 × study_hours
   + b2 × attendance
   + b3 × previous_score

3. Apply the sigmoid function.

4. Obtain Pass Probability.

5. Apply threshold = 0.5.

   Probability >= 0.5
   → Predict Pass (1)

   Probability < 0.5
   → Predict Fail (0)

6. Compare prediction with the actual result.
"""
)


# ------------------------------------------------------------
# 17. IMPORTANT NOTE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    """
The model never used these test students
while learning b0, b1, b2 and b3.

That is why these predictions help us
evaluate how the trained model performs
on unseen data.
"""
)


# ------------------------------------------------------------
# 18. SAVE PREDICTIONS
# ------------------------------------------------------------

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# 19. DISPLAY SAVED FILE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PREDICTIONS SAVED")
print("=" * 80)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 20. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 80)