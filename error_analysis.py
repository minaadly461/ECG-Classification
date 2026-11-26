
import numpy as np
import pandas as pd

# =========================
# 1. Load Results
# =========================

data = np.load("models/ensemble_test_results.npz")

y_true = data["y_true"]
y_pred = data["predictions"]
probabilities = data["probabilities"]

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

# =========================
# 2. Analyze Errors
# =========================

all_errors = []

print("=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

for i, class_name in enumerate(CLASS_NAMES):

    actual = y_true[:, i]
    predicted = y_pred[:, i]
    probs = probabilities[:, i]

    fp_indices = np.where(
        (actual == 0) & (predicted == 1)
    )[0]

    fn_indices = np.where(
        (actual == 1) & (predicted == 0)
    )[0]

    tp = np.sum((actual == 1) & (predicted == 1))
    tn = np.sum((actual == 0) & (predicted == 0))

    print(f"\n{class_name}")
    print("-" * 40)
    print(f"True Positives : {tp}")
    print(f"True Negatives : {tn}")
    print(f"False Positives: {len(fp_indices)}")
    print(f"False Negatives: {len(fn_indices)}")

    # Store False Positives
    for idx in fp_indices:
        all_errors.append({
            "index": idx,
            "class": class_name,
            "error_type": "False Positive",
            "probability": probs[idx],
            "actual": actual[idx],
            "predicted": predicted[idx]
        })

    # Store False Negatives
    for idx in fn_indices:
        all_errors.append({
            "index": idx,
            "class": class_name,
            "error_type": "False Negative",
            "probability": probs[idx],
            "actual": actual[idx],
            "predicted": predicted[idx]
        })

# =========================
# 3. Save Error Details
# =========================

errors_df = pd.DataFrame(all_errors)

errors_df.to_csv(
    "models/error_analysis.csv",
    index=False
)

print("\nError details saved to:")
print("models/error_analysis.csv")

# =========================
# 4. Most Confident Errors
# =========================

print("\n" + "=" * 60)
print("TOP 10 MOST CONFIDENT ERRORS PER CLASS")
print("=" * 60)

for class_name in CLASS_NAMES:

    class_errors = errors_df[
        errors_df["class"] == class_name
    ].copy()

    class_errors["confidence"] = np.where(
        class_errors["error_type"] == "False Positive",
        class_errors["probability"],
        1 - class_errors["probability"]
    )

    top_errors = class_errors.sort_values(
        "confidence",
        ascending=False
    ).head(10)

    print(f"\n{class_name}:")
    print(
        top_errors[
            ["index", "error_type", "probability", "confidence"]
        ].to_string(index=False)
    )

print("\nError analysis completed!")


import matplotlib.pyplot as plt

# Select the most confident errors
selected_indices = [1040, 2083, 182, 891, 526]

# Load test signals
data = np.load("data/processed/ecg_dataset.npz")
X_test = data["X_test"]

for idx in selected_indices:

    fig, axes = plt.subplots(
        12, 1,
        figsize=(15, 18),
        sharex=True
    )

    for lead in range(12):
        axes[lead].plot(X_test[idx, :, lead])
        axes[lead].set_ylabel(f"Lead {lead+1}")
        axes[lead].grid(alpha=0.3)

    actual = [
    CLASS_NAMES[i]
    for i in np.where(y_true[idx] == 1)[0]
]

    predicted = [
        CLASS_NAMES[i]
        for i in np.where(y_pred[idx] == 1)[0]
    ]

    plt.suptitle(
        f"ECG Index: {idx}\n"
        f"Actual: {actual} | Predicted: {predicted}",
        fontsize=13
    )

    plt.tight_layout()
    plt.show()
