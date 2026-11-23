import json
import os
import shutil
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 42
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

df = pd.read_csv("dataset/diabetes.csv")
print("shape:", df.shape)
print(df["Outcome"].value_counts(), "\n")

# glucose/bmi etc can't actually be 0, those are just missing readings
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    print(f"{col}: {(df[col] == 0).sum()} zeros -> NaN")
    df[col] = df[col].replace(0, np.nan)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# fill using train medians only so nothing leaks from the test set
medians = X_train[zero_cols].median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=SEED),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = []

for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)[:, 1]
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "CV Accuracy (mean)": cv_scores.mean(),
        "CV Accuracy (std)": cv_scores.std(),
    })
    print(f"{name:25s} acc={results[-1]['Accuracy']:.3f}  cv={cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# test set is only ~150 rows so pick the winner on cv score instead
results_df = pd.DataFrame(results).sort_values("CV Accuracy (mean)", ascending=False).reset_index(drop=True)
print("\n", results_df.to_string(index=False))

best_name = results_df.loc[0, "Model"]
best_model = models[best_name]
print(f"\nbest: {best_name}")

results_df.to_csv("models/diabetes_model_comparison.csv", index=False)
joblib.dump(best_model, "models/diabetes_best_model.pkl")
joblib.dump(scaler, "models/diabetes_scaler.pkl")
joblib.dump(list(X.columns), "models/diabetes_feature_names.pkl")

with open("models/diabetes_model_info.json", "w") as f:
    json.dump({
        "best_model_name": best_name,
        "accuracy": round(float(results_df.loc[0, "Accuracy"]), 4),
        "cv_accuracy": round(float(results_df.loc[0, "CV Accuracy (mean)"]), 4),
        "fill_values": {col: round(float(v), 2) for col, v in medians.items()},
        "all_results": results_df.to_dict(orient="records"),
    }, f, indent=2)

sns.set_style("whitegrid")

plt.figure(figsize=(9, 5))
colors = ["#06A77D" if m == best_name else "#2E86AB" for m in results_df["Model"]]
bars = plt.bar(results_df["Model"], results_df["CV Accuracy (mean)"], color=colors)
for bar, acc in zip(bars, results_df["CV Accuracy (mean)"]):
    plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=9)
plt.ylabel("5-Fold CV Accuracy")
plt.title("Diabetes Prediction: Model Comparison (5-Fold CV Accuracy)")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/diabetes_model_comparison.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, best_model.predict(X_test_s))
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("plots/diabetes_confusion_matrix.png", dpi=150)
plt.close()

if hasattr(best_model, "feature_importances_"):
    importance = best_model.feature_importances_
    xlabel = "Importance"
elif hasattr(best_model, "coef_"):
    importance = np.abs(best_model.coef_[0])
    xlabel = "Absolute Coefficient Value (impact on prediction)"
else:
    # knn / svm don't expose anything, fall back to permutation importance
    perm = permutation_importance(best_model, X_test_s, y_test, scoring="roc_auc",
                                  n_repeats=30, random_state=SEED)
    importance = perm.importances_mean
    xlabel = "Drop in ROC-AUC when the feature is shuffled"

imp = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values("Importance")

plt.figure(figsize=(8, 5))
plt.barh(imp["Feature"], imp["Importance"], color="#06A77D")
plt.xlabel(xlabel)
plt.title(f"What matters most for Diabetes Prediction? ({best_name})")
plt.tight_layout()
plt.savefig("plots/diabetes_feature_importance.png", dpi=150)
plt.close()

# flask serves from static/, so copy the charts over for the insights page
os.makedirs("static/plots", exist_ok=True)
for chart in ["diabetes_model_comparison.png", "diabetes_confusion_matrix.png", "diabetes_feature_importance.png"]:
    shutil.copy(f"plots/{chart}", f"static/plots/{chart}")

print("done, models saved to models/ and charts to plots/")
