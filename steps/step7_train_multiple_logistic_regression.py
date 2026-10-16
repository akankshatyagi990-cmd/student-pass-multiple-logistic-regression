import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 7 - TRAIN MULTIPLE LOGISTIC REGRESSION
# USING GRADIENT DESCENT
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step6_scaled_train_data.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step7_model_parameters.csv"
)

HISTORY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step7_training_history.csv"
)


# ------------------------------------------------------------
# 2. LOAD SCALED TRAINING DATA
# ------------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)


# ------------------------------------------------------------
# 3. SELECT INPUT FEATURES
# ------------------------------------------------------------

feature_names = [
    "study_hours",
    "attendance",
    "previous_score",
]


X = train_df[
    feature_names
].to_numpy(dtype=float)


# ------------------------------------------------------------
# 4. SELECT TARGET
# ------------------------------------------------------------

y = train_df[
    "passed"
].to_numpy(dtype=float)


# ------------------------------------------------------------
# 5. DISPLAY DATA SHAPE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TRAINING DATA")
print("=" * 80)

print(
    f"Training Students : {X.shape[0]}"
)

print(
    f"Number of Features: {X.shape[1]}"
)


print("\nFeatures:")

for feature in feature_names:
    print(
        f"- {feature}"
    )


# ------------------------------------------------------------
# 6. CREATE SIGMOID FUNCTION
# ------------------------------------------------------------

def sigmoid(z):

    # Prevent extremely large values
    # from causing numerical problems.

    z = np.clip(
        z,
        -500,
        500,
    )

    return 1 / (
        1 + np.exp(-z)
    )


# ------------------------------------------------------------
# 7. INITIALIZE MODEL PARAMETERS
# ------------------------------------------------------------

b0 = 0.0


# Three input features
# therefore three coefficients.

weights = np.zeros(
    X.shape[1],
    dtype=float,
)


print("\n" + "=" * 80)
print("INITIAL PARAMETERS")
print("=" * 80)

print(
    f"b0 = {b0}"
)

print(
    f"b1 = {weights[0]}"
)

print(
    f"b2 = {weights[1]}"
)

print(
    f"b3 = {weights[2]}"
)


# ------------------------------------------------------------
# 8. SET TRAINING SETTINGS
# ------------------------------------------------------------

learning_rate = 0.1

epochs = 10000


print("\n" + "=" * 80)
print("TRAINING SETTINGS")
print("=" * 80)

print(
    f"Learning Rate : {learning_rate}"
)

print(
    f"Epochs        : {epochs}"
)


# ------------------------------------------------------------
# 9. PREPARE TRAINING HISTORY
# ------------------------------------------------------------

history = []


# ------------------------------------------------------------
# 10. TRAIN MODEL USING GRADIENT DESCENT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TRAINING MODEL")
print("=" * 80)


for epoch in range(
    1,
    epochs + 1,
):


    # --------------------------------------------------------
    # STEP A - CALCULATE z
    # --------------------------------------------------------

    z = (
        b0
        + np.dot(
            X,
            weights,
        )
    )


    # --------------------------------------------------------
    # STEP B - APPLY SIGMOID
    # --------------------------------------------------------

    probabilities = sigmoid(
        z
    )


    # --------------------------------------------------------
    # STEP C - CALCULATE LOG LOSS
    # --------------------------------------------------------

    epsilon = 1e-15

    safe_probabilities = np.clip(
        probabilities,
        epsilon,
        1 - epsilon,
    )


    loss = -np.mean(
        y * np.log(
            safe_probabilities
        )
        +
        (1 - y)
        * np.log(
            1 - safe_probabilities
        )
    )


    # --------------------------------------------------------
    # STEP D - CALCULATE ERRORS
    # --------------------------------------------------------

    errors = (
        probabilities
        - y
    )


    # --------------------------------------------------------
    # STEP E - GRADIENT FOR INTERCEPT
    # --------------------------------------------------------

    db0 = np.mean(
        errors
    )


    # --------------------------------------------------------
    # STEP F - GRADIENTS FOR ALL THREE COEFFICIENTS
    # --------------------------------------------------------

    dw = (
        np.dot(
            X.T,
            errors,
        )
        / len(y)
    )


    # --------------------------------------------------------
    # STEP G - UPDATE PARAMETERS
    # --------------------------------------------------------

    b0 = (
        b0
        - learning_rate
        * db0
    )


    weights = (
        weights
        - learning_rate
        * dw
    )


    # --------------------------------------------------------
    # STEP H - SAVE TRAINING HISTORY
    # --------------------------------------------------------

    history.append(
        {
            "epoch": epoch,
            "loss": loss,
            "b0": b0,
            "b1_study_hours": weights[0],
            "b2_attendance": weights[1],
            "b3_previous_score": weights[2],
        }
    )


    # --------------------------------------------------------
    # STEP I - PRINT IMPORTANT EPOCHS
    # --------------------------------------------------------

    if epoch in [
        1,
        10,
        100,
        500,
        1000,
        5000,
        10000,
    ]:

        print(
            f"\nEpoch {epoch}"
        )

        print(
            f"Loss = {loss:.6f}"
        )

        print(
            f"b0 = {b0:.6f}"
        )

        print(
            f"b1 = {weights[0]:.6f}"
        )

        print(
            f"b2 = {weights[1]:.6f}"
        )

        print(
            f"b3 = {weights[2]:.6f}"
        )


# ------------------------------------------------------------
# 11. DISPLAY FINAL MODEL PARAMETERS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL MODEL PARAMETERS")
print("=" * 80)

print(
    f"b0 = {b0:.6f}"
)

print(
    f"b1 = {weights[0]:.6f}  "
    "(study_hours)"
)

print(
    f"b2 = {weights[1]:.6f}  "
    "(attendance)"
)

print(
    f"b3 = {weights[2]:.6f}  "
    "(previous_score)"
)


# ------------------------------------------------------------
# 12. DISPLAY MODEL EQUATION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LEARNED MODEL EQUATION")
print("=" * 80)

print(
    f"""
z =
{b0:.6f}
+ ({weights[0]:.6f} × scaled_study_hours)
+ ({weights[1]:.6f} × scaled_attendance)
+ ({weights[2]:.6f} × scaled_previous_score)
"""
)


# ------------------------------------------------------------
# 13. INTERPRET COEFFICIENT DIRECTIONS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("COEFFICIENT INTERPRETATION")
print("=" * 80)


for feature, coefficient in zip(
    feature_names,
    weights,
):

    if coefficient > 0:

        direction = (
            "increases the probability of passing"
        )

    elif coefficient < 0:

        direction = (
            "decreases the probability of passing"
        )

    else:

        direction = (
            "has no learned effect"
        )


    print(
        f"{feature}: "
        f"{coefficient:.6f} "
        f"→ {direction}"
    )


# ------------------------------------------------------------
# 14. CREATE MODEL PARAMETER TABLE
# ------------------------------------------------------------

model_parameters = pd.DataFrame(
    {
        "parameter": [
            "b0",
            "b1_study_hours",
            "b2_attendance",
            "b3_previous_score",
            "learning_rate",
            "epochs",
        ],

        "value": [
            b0,
            weights[0],
            weights[1],
            weights[2],
            learning_rate,
            epochs,
        ],
    }
)


# ------------------------------------------------------------
# 15. CREATE TRAINING HISTORY TABLE
# ------------------------------------------------------------

history_df = pd.DataFrame(
    history
)


# ------------------------------------------------------------
# 16. SAVE MODEL PARAMETERS
# ------------------------------------------------------------

model_parameters.to_csv(
    MODEL_FILE,
    index=False,
)


# ------------------------------------------------------------
# 17. SAVE TRAINING HISTORY
# ------------------------------------------------------------

history_df.to_csv(
    HISTORY_FILE,
    index=False,
)


# ------------------------------------------------------------
# 18. EXPLAIN GRADIENT DESCENT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHAT DID GRADIENT DESCENT DO?")
print("=" * 80)

print(
    """
We started with:

b0 = 0
b1 = 0
b2 = 0
b3 = 0


During every epoch:

1. Calculate z.

2. Apply the sigmoid function.

3. Calculate predicted probabilities.

4. Compare probabilities with actual results.

5. Calculate Log Loss.

6. Calculate gradients.

7. Update b0, b1, b2 and b3.

8. Repeat the process.


The goal was to reduce Log Loss
and find better model coefficients.
"""
)


# ------------------------------------------------------------
# 19. EXPLAIN THE IMPORTANT DIFFERENCE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("IMPORTANT DIFFERENCE FROM PREVIOUS PROJECT")
print("=" * 80)

print(
    """
In our previous Logistic Regression project,
we had only one input feature.

Therefore the decision boundary could be
described using one study-hours value.


Now we have THREE features:

study_hours
attendance
previous_score


So there is no single study-hours decision
boundary such as:

4.14 hours


Instead, the decision depends on the
COMBINATION of all three features.

The decision boundary occurs where:

z = 0

because:

sigmoid(0) = 0.5
"""
)


# ------------------------------------------------------------
# 20. DISPLAY SAVED FILES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Model Parameters : {MODEL_FILE}"
)

print(
    f"Training History : {HISTORY_FILE}"
)


# ------------------------------------------------------------
# 21. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 80)