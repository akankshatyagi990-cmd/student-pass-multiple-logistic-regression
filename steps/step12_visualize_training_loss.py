import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 12 - VISUALIZE TRAINING LOSS
# MULTIPLE LOGISTIC REGRESSION
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step7_training_history.csv"
)

PLOT_FILE = (
    PROJECT_ROOT
    / "plots"
    / "step12_training_loss.png"
)


# ------------------------------------------------------------
# 2. LOAD TRAINING HISTORY
# ------------------------------------------------------------

history_df = pd.read_csv(
    HISTORY_FILE
)


# ------------------------------------------------------------
# 3. DISPLAY FIRST FEW ROWS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FIRST 10 TRAINING HISTORY ROWS")
print("=" * 80)

print(
    history_df.head(10)
)


# ------------------------------------------------------------
# 4. DISPLAY LAST FEW ROWS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LAST 10 TRAINING HISTORY ROWS")
print("=" * 80)

print(
    history_df.tail(10)
)


# ------------------------------------------------------------
# 5. GET INITIAL LOSS
# ------------------------------------------------------------

initial_loss = history_df.loc[
    0,
    "loss"
]


# ------------------------------------------------------------
# 6. GET FINAL LOSS
# ------------------------------------------------------------

final_loss = history_df.loc[
    len(history_df) - 1,
    "loss"
]


# ------------------------------------------------------------
# 7. CALCULATE LOSS REDUCTION
# ------------------------------------------------------------

loss_reduction = (
    initial_loss
    - final_loss
)


loss_reduction_percentage = (
    loss_reduction
    / initial_loss
    * 100
)


# ------------------------------------------------------------
# 8. DISPLAY LOSS SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LOSS SUMMARY")
print("=" * 80)

print(
    f"Initial Loss : {initial_loss:.6f}"
)

print(
    f"Final Loss   : {final_loss:.6f}"
)

print(
    f"Loss Reduced : {loss_reduction:.6f}"
)

print(
    f"Reduction    : {loss_reduction_percentage:.2f}%"
)


# ------------------------------------------------------------
# 9. DISPLAY SELECTED EPOCHS
# ------------------------------------------------------------

important_epochs = [
    1,
    10,
    100,
    500,
    1000,
    5000,
    10000,
]


selected_history = history_df[
    history_df["epoch"].isin(
        important_epochs
    )
]


print("\n" + "=" * 80)
print("LOSS AT IMPORTANT EPOCHS")
print("=" * 80)

print(
    selected_history[
        [
            "epoch",
            "loss",
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 10. CREATE LOSS CURVE
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    history_df["epoch"],
    history_df["loss"],
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Log Loss"
)

plt.title(
    "Training Loss - Multiple Logistic Regression"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ------------------------------------------------------------
# 11. SAVE LOSS CURVE
# ------------------------------------------------------------

plt.savefig(
    PLOT_FILE,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ------------------------------------------------------------
# 12. CHECK WHETHER LOSS DECREASED
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TRAINING CHECK")
print("=" * 80)


if final_loss < initial_loss:

    print(
        "SUCCESS: Log Loss decreased during training."
    )

else:

    print(
        "WARNING: Log Loss did not decrease."
    )


# ------------------------------------------------------------
# 13. EXPLAIN THE GRAPH
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("HOW TO READ THE GRAPH")
print("=" * 80)

print(
    """
X-axis:
Epoch number

Y-axis:
Log Loss


At the beginning:

The coefficients are not yet well trained,
so the loss is higher.


As Gradient Descent runs:

b0
b1
b2
b3

are updated repeatedly.


If training is working correctly:

Loss should generally move downward.


Eventually the curve should become flatter.

That means the model is getting close to
a point where additional epochs produce
only very small improvements.
"""
)


# ------------------------------------------------------------
# 14. EXPLAIN CONVERGENCE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WHAT IS CONVERGENCE?")
print("=" * 80)

print(
    """
Convergence means the model has reached
a stage where its parameters are changing
only slightly and the loss is no longer
decreasing significantly.

In simple terms:

Early epochs
→ Big learning changes

Later epochs
→ Smaller learning changes

Eventually
→ Loss becomes almost stable
"""
)


# ------------------------------------------------------------
# 15. IMPORTANT NOTE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    """
More epochs are not automatically better.

If the loss has already become nearly stable,
running many more epochs may provide very
little additional improvement.

We use the loss curve to understand whether
Gradient Descent is learning and whether the
training process is approaching convergence.
"""
)


# ------------------------------------------------------------
# 16. DISPLAY SAVED FILE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PLOT SAVED")
print("=" * 80)

print(
    PLOT_FILE
)


# ------------------------------------------------------------
# 17. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 12 COMPLETED SUCCESSFULLY")
print("=" * 80)