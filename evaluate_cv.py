
import numpy as np

# Load cross-validation results
results = np.load("models/cnn_cv_results.npy")

auc_scores = results[:, 0]
f1_scores = results[:, 1]

print("=" * 45)
print("5-FOLD CROSS-VALIDATION EVALUATION")
print("=" * 45)

for i in range(len(results)):
    print(
        f"Fold {i+1}: "
        f"AUC = {auc_scores[i]:.4f} | "
        f"F1 = {f1_scores[i]:.4f}"
    )

print("\nFinal Results")
print("-" * 45)

print(
    f"Mean ROC-AUC: "
    f"{auc_scores.mean():.4f} ± {auc_scores.std():.4f}"
)

print(
    f"Mean Macro F1: "
    f"{f1_scores.mean():.4f} ± {f1_scores.std():.4f}"
)

print("\nStability")
print("-" * 45)

print(f"AUC Range: {auc_scores.min():.4f} - {auc_scores.max():.4f}")
print(f"F1 Range: {f1_scores.min():.4f} - {f1_scores.max():.4f}")