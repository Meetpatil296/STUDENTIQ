"""
train_model.py
Loads the dataset, analyses it, cleans it, builds a leakage-free
preprocessing + Random Forest pipeline, evaluates it on unseen test data
and saves everything needed by the Streamlit app.

Run:  python train_model.py
"""

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")                       # draw graphs without a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import (CATEGORICAL_FEATURES, CLASS_ORDER, FEATURES, LABELS,
                    METRICS_PATH, MODEL_PATH, NUMERIC_FEATURES, REPORT_DIR,
                    TARGET, clean_dataset, load_dataset)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------
# STEP 1 - Dataset analysis (before cleaning)
# ----------------------------------------------------------------------
def analyse_raw_data(df):
    lines = []
    lines.append("=" * 70)
    lines.append("DATASET ANALYSIS (RAW DATA)")
    lines.append("=" * 70)
    lines.append(f"Total rows: {df.shape[0]}   Total columns: {df.shape[1]}\n")
    lines.append("First 10 records:\n" + df.head(10).to_string(index=False) + "\n")
    lines.append("Column data types:\n" + df.dtypes.to_string() + "\n")
    lines.append("Statistical summary (numeric columns):\n" + df.describe().round(2).to_string() + "\n")
    lines.append("Missing values per column:\n" + df.isna().sum().to_string() + "\n")
    lines.append(f"Duplicate records (exact copies): {df.duplicated().sum()}\n")
    lines.append("Target class distribution:\n" + df[TARGET].value_counts().to_string())
    text = "\n".join(lines)
    print(text)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "dataset_analysis.txt").write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------
# STEP 2 - Exploratory Data Analysis graphs
# ----------------------------------------------------------------------
def save_fig(name):
    plt.tight_layout()
    plt.savefig(REPORT_DIR / name, dpi=120)
    plt.close()


def make_eda_plots(df):
    palette = {"Poor": "#e76f51", "Average": "#f4a261", "Good": "#2a9d8f"}

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=TARGET, hue=TARGET, order=CLASS_ORDER, palette=palette, legend=False)
    ax.bar_label(ax.containers[0])
    plt.title("Distribution of Student Performance Categories")
    plt.xlabel("Performance category"); plt.ylabel("Number of students")
    save_fig("eda_class_distribution.png")

    pairs = [("Study_Hours_Per_Day", "eda_study_vs_performance.png"),
             ("Attendance_Percentage", "eda_attendance_vs_performance.png"),
             ("Previous_Exam_Marks", "eda_previous_marks_vs_performance.png"),
             ("Assignment_Scores", "eda_assignment_vs_performance.png")]
    for col, fname in pairs:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, order=CLASS_ORDER, palette=palette, legend=False)
        plt.title(f"{LABELS[col]} vs Performance")
        plt.xlabel("Performance category"); plt.ylabel(LABELS[col])
        save_fig(fname)

    plt.figure(figsize=(6, 4))
    sns.histplot(df["Sleep_Hours"], bins=20, kde=True, color="#457b9d")
    plt.title("Distribution of Sleep Hours")
    plt.xlabel("Sleep hours per day"); plt.ylabel("Number of students")
    save_fig("eda_sleep_distribution.png")

    corr_df = df[NUMERIC_FEATURES].copy()
    corr_df["Performance (Poor=0, Avg=1, Good=2)"] = df[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)})
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    save_fig("eda_correlation_heatmap.png")


# ----------------------------------------------------------------------
# STEP 3 - Pipeline: preprocessing + Random Forest
# ----------------------------------------------------------------------
def build_pipeline():
    numeric_steps = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_steps, NUMERIC_FEATURES),
        ("cat", categorical_steps, CATEGORICAL_FEATURES),
    ])
    # Random Forest does not need feature scaling (trees split on thresholds)
    model = RandomForestClassifier(
        n_estimators=200,        # number of decision trees
        max_depth=10,            # limit tree depth to reduce overfitting
        min_samples_leaf=5,      # each leaf must hold at least 5 students
        max_features="sqrt",     # random subset of features at each split
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def plot_confusion_matrix(cm):
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
    plt.title("Confusion Matrix (Test Data)")
    plt.xlabel("Predicted category"); plt.ylabel("Actual category")
    save_fig("confusion_matrix.png")


def plot_feature_importance(importances):
    imp = pd.Series(importances).sort_values()
    plt.figure(figsize=(7, 4.5))
    imp.plot(kind="barh", color="#2a9d8f")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance"); plt.ylabel("Feature")
    save_fig("feature_importance.png")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    raw = load_dataset()
    analyse_raw_data(raw)

    print("\n" + "=" * 70 + "\nPREPROCESSING\n" + "=" * 70)
    df = clean_dataset(raw)
    print(f"Rows before cleaning: {len(raw)}  ->  after removing duplicates: {len(df)}")
    print("Missing feature values are filled INSIDE the pipeline "
          "(median / most frequent), fitted on training data only.")

    make_eda_plots(df)
    print("EDA graphs saved in the 'reports' folder.")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print(f"Training rows: {len(X_train)}   Testing rows: {len(X_test)}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # --- Evaluation on unseen test data ---
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    cv = cross_val_score(build_pipeline(), X_train, y_train,
                         cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                         scoring="accuracy")

    feature_names = [n.split("__", 1)[1] for n in
                     pipeline.named_steps["preprocess"].get_feature_names_out()]
    importances = dict(zip(feature_names, pipeline.named_steps["model"].feature_importances_.round(4).tolist()))

    metrics = {
        "dataset": {"raw_rows": int(len(raw)), "clean_rows": int(len(df)),
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "train_accuracy": float(accuracy_score(y_train, pipeline.predict(X_train))),
        "cv_accuracy_mean": float(cv.mean()),
        "cv_accuracy_std": float(cv.std()),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classes": CLASS_ORDER,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importance": importances,
        "hyperparameters": {k: str(v) for k, v in pipeline.named_steps["model"].get_params().items()
                            if k in ("n_estimators", "max_depth", "min_samples_leaf",
                                     "max_features", "class_weight", "random_state")},
    }

    plot_confusion_matrix(cm)
    plot_feature_importance(importances)

    print("\n" + "=" * 70 + "\nMODEL EVALUATION (test data)\n" + "=" * 70)
    print(f"Accuracy             : {metrics['accuracy']:.4f}")
    print(f"Precision (macro)    : {metrics['precision_macro']:.4f}")
    print(f"Recall (macro)       : {metrics['recall_macro']:.4f}")
    print(f"F1-score (macro)     : {metrics['f1_macro']:.4f}")
    print(f"Training accuracy    : {metrics['train_accuracy']:.4f}  (compare with test accuracy)")
    print(f"5-fold CV accuracy   : {metrics['cv_accuracy_mean']:.4f} +/- {metrics['cv_accuracy_std']:.4f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))
    print("\nClassification report:\n" + classification_report(y_test, y_pred, labels=CLASS_ORDER, zero_division=0))
    print("Feature importance:")
    print(pd.Series(importances).sort_values(ascending=False).to_string())

    # --- Save model and metrics ---
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModel saved   : {MODEL_PATH}")
    print(f"Metrics saved : {METRICS_PATH}")


if __name__ == "__main__":
    main()
