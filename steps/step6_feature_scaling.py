import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 6 - FEATURE SCALING
# MULTIPLE LOGISTIC REGRESSION
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

SCALED_TRAIN_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step6_scaled_train_data.csv"
)

SCALED_TEST_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step6_scaled_test_data.csv"
)

SCALER_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step6_scaler_parameters.csv"
)


# ------------------------------------------------------------
# 2. LOAD TRAINING AND TEST DATA
# ------------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)

test_df = pd.read_csv(TEST_FILE)


# ------------------------------------------------------------
# 3. DEFINE FEATURES TO SCALE
# ------------------------------------------------------------

features = [
    "study_hours",
    "attendance",
    "previous_score",
]


# ------------------------------------------------------------
# 4. DISPLAY ORIGINAL TRAINING FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ORIGINAL TRAINING FEATURES")
print("=" * 80)

print(
    train_df[features].head(10)
)


# ------------------------------------------------------------
# 5. CALCULATE MEAN FROM TRAINING DATA ONLY
# ------------------------------------------------------------

training_means = (
    train_df[features]
    .mean()
)


# ------------------------------------------------------------
# 6. CALCULATE STANDARD DEVIATION FROM TRAINING DATA ONLY
# ------------------------------------------------------------

# ddof=0 gives population standard deviation.
# This is also the convention used by StandardScaler.

training_stds = (
    train_df[features]
    .std(ddof=0)
)


# ------------------------------------------------------------
# 7. DISPLAY SCALING PARAMETERS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TRAINING MEANS")
print("=" * 80)

print(
    training_means
)


print("\n" + "=" * 80)
print("TRAINING STANDARD DEVIATIONS")
print("=" * 80)

print(
    training_stds
)


# ------------------------------------------------------------
# 8. CREATE COPIES BEFORE SCALING
# ------------------------------------------------------------

scaled_train_df = train_df.copy()

scaled_test_df = test_df.copy()


# ------------------------------------------------------------
# 9. SCALE TRAINING FEATURES
# ------------------------------------------------------------

for feature in features:

    scaled_train_df[feature] = (
        train_df[feature]
        - training_means[feature]
    ) / training_stds[feature]


# ------------------------------------------------------------
# 10. SCALE TEST FEATURES
# ------------------------------------------------------------

# IMPORTANT:
# We use the TRAINING mean and TRAINING standard deviation.
# We do NOT calculate new values from the test data.

for feature in features:

    scaled_test_df[feature] = (
        test_df[feature]
        - training_means[feature]
    ) / training_stds[feature]


# ------------------------------------------------------------
# 11. DISPLAY SCALED TRAINING DATA
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SCALED TRAINING FEATURES")
print("=" * 80)

print(
    scaled_train_df[features].head(10)
)


# ------------------------------------------------------------
# 12. DISPLAY SCALED TEST DATA
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SCALED TEST FEATURES")
print("=" * 80)

print(
    scaled_test_df[features]
)


# ------------------------------------------------------------
# 13. CHECK SCALED TRAINING MEANS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SCALED TRAINING MEANS")
print("=" * 80)

print(
    scaled_train_df[features]
    .mean()
)


# ------------------------------------------------------------
# 14. CHECK SCALED TRAINING STANDARD DEVIATIONS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SCALED TRAINING STANDARD DEVIATIONS")
print("=" * 80)

print(
    scaled_train_df[features]
    .std(ddof=0)
)


# ------------------------------------------------------------
# 15. SHOW ONE EXAMPLE MANUALLY
# ------------------------------------------------------------

example_feature = "study_hours"

original_value = train_df.loc[
    0,
    example_feature
]

mean_value = training_means[
    example_feature
]

std_value = training_stds[
    example_feature
]

scaled_value = (
    original_value
    - mean_value
) / std_value


print("\n" + "=" * 80)
print("ONE SCALING EXAMPLE")
print("=" * 80)

print(
    f"Feature            : {example_feature}"
)

print(
    f"Original Value     : {original_value:.6f}"
)

print(
    f"Training Mean      : {mean_value:.6f}"
)

print(
    f"Training Std       : {std_value:.6f}"
)

print(
    "\nFormula:"
)

print(
    "scaled = "
    "(original - mean) / standard_deviation"
)

print(
    "\nCalculation:"
)

print(
    f"scaled = "
    f"({original_value:.6f} - {mean_value:.6f}) "
    f"/ {std_value:.6f}"
)

print(
    f"\nScaled Value       : {scaled_value:.6f}"
)


# ------------------------------------------------------------
# 16. CREATE SCALER PARAMETER TABLE
# ------------------------------------------------------------

scaler_parameters = pd.DataFrame(
    {
        "feature": features,

        "mean": [
            training_means[feature]
            for feature in features
        ],

        "standard_deviation": [
            training_stds[feature]
            for feature in features
        ],
    }
)


# ------------------------------------------------------------
# 17. SAVE SCALED TRAINING DATA
# ------------------------------------------------------------

scaled_train_df.to_csv(
    SCALED_TRAIN_FILE,
    index=False,
)


# ------------------------------------------------------------
# 18. SAVE SCALED TEST DATA
# ------------------------------------------------------------

scaled_test_df.to_csv(
    SCALED_TEST_FILE,
    index=False,
)


# ------------------------------------------------------------
# 19. SAVE SCALER PARAMETERS
# ------------------------------------------------------------

scaler_parameters.to_csv(
    SCALER_FILE,
    index=False,
)


# ------------------------------------------------------------
# 20. EXPLAIN WHAT HAPPENED
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHAT DID WE DO?")
print("=" * 80)

print(
    """
We standardized our three input features:

study_hours
attendance
previous_score


For every feature we used:

scaled value
=
(original value - training mean)
/
training standard deviation


After scaling the TRAINING data:

Mean should be approximately 0.

Standard deviation should be approximately 1.


This gives the features a more comparable scale.
"""
)


# ------------------------------------------------------------
# 21. EXPLAIN NEGATIVE AND POSITIVE VALUES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHAT DO THE SCALED VALUES MEAN?")
print("=" * 80)

print(
    """
A scaled value around:

0
→ close to the training average

Positive value
→ above the training average

Negative value
→ below the training average


For example:

study_hours = -1.2

does NOT mean negative study hours.

It means the student's study hours are
below the average study hours in the
training dataset.
"""
)


# ------------------------------------------------------------
# 22. EXPLAIN DATA LEAKAGE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHY TRAINING DATA ONLY?")
print("=" * 80)

print(
    """
We calculated the mean and standard deviation
using ONLY the training data.

Then we used those SAME values to transform
the test data.

We did not allow the test dataset to influence
how the model data was prepared.

This helps prevent:

DATA LEAKAGE
"""
)


# ------------------------------------------------------------
# 23. DISPLAY SAVED FILES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Scaled Training Data : {SCALED_TRAIN_FILE}"
)

print(
    f"Scaled Test Data     : {SCALED_TEST_FILE}"
)

print(
    f"Scaler Parameters    : {SCALER_FILE}"
)


# ------------------------------------------------------------
# 24. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 80)