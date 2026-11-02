"""
generate_dataset.py
Creates a SYNTHETIC (computer generated) student dataset and saves it as
dataset/student_performance.csv

The dataset is NOT real student data. Values are produced from a hidden
"academic ability" factor plus random noise, so features are realistically
related to the final category, but with overlap between categories
(some hard-working students still perform averagely, etc.).

Run:  python generate_dataset.py
"""

import numpy as np
import pandas as pd

from common import DATA_PATH, TARGET

SEED = 42
N_STUDENTS = 1800        # unique students
N_DUPLICATES = 40        # exact duplicate rows added on purpose
MISSING_RATE = 0.02      # ~2 % of values missing in selected columns


def zscore(x):
    return (x - x.mean()) / x.std()


def generate(n=N_STUDENTS, seed=SEED):
    rng = np.random.default_rng(seed)

    # Hidden factor: overall ability / diligence of each student
    ability = rng.normal(0, 1, n)

    study = np.clip(3.3 + 0.9 * ability + rng.normal(0, 1.0, n), 0.5, 9.0)
    attendance = np.clip(78 + 7.5 * ability + rng.normal(0, 8.0, n), 40, 100)
    previous = np.clip(62 + 12 * ability + rng.normal(0, 9.0, n), 20, 100)
    assignment = np.clip(66 + 10 * ability + rng.normal(0, 10.0, n), 20, 100)
    internal = np.clip(63 + 11 * ability + rng.normal(0, 9.0, n), 15, 100)
    sleep = np.clip(rng.normal(7.0, 1.0, n) + 0.1 * ability, 4.0, 10.0)
    extra = rng.random(n) < 0.45
    internet = rng.random(n) < 1 / (1 + np.exp(-(1.2 + 0.3 * ability)))

    # Healthy sleep (6.5 - 8.5 h) helps a little; too little/too much hurts
    sleep_effect = -np.abs(sleep - 7.5)

    # Final performance score = weighted mix of the features + noise
    score = (
        0.30 * zscore(previous)
        + 0.20 * zscore(internal)
        + 0.15 * zscore(assignment)
        + 0.15 * zscore(attendance)
        + 0.12 * zscore(study)
        + 0.04 * zscore(sleep_effect)
        + 0.04 * internet
        + 0.02 * extra
        + rng.normal(0, 0.40, n)                 # natural variation
    )
    # 30 % Poor, 40 % Average, 30 % Good (reasonably balanced)
    low, high = np.quantile(score, [0.30, 0.70])
    category = np.where(score < low, "Poor", np.where(score < high, "Average", "Good"))

    df = pd.DataFrame(
        {
            "Student_ID": [f"STU{i:04d}" for i in range(1, n + 1)],
            "Study_Hours_Per_Day": study.round(1),
            "Attendance_Percentage": attendance.round(1),
            "Previous_Exam_Marks": previous.round(1),
            "Assignment_Scores": assignment.round(1),
            "Internal_Assessment_Marks": internal.round(1),
            "Sleep_Hours": sleep.round(1),
            "Extracurricular_Activity": np.where(extra, "Yes", "No"),
            "Internet_Availability": np.where(internet, "Yes", "No"),
            TARGET: category,
        }
    )

    # --- Introduce a few missing values (to demonstrate preprocessing) ---
    for col in ["Study_Hours_Per_Day", "Attendance_Percentage", "Assignment_Scores",
                "Sleep_Hours", "Extracurricular_Activity", "Internet_Availability"]:
        mask = rng.random(n) < MISSING_RATE
        df[col] = df[col].astype(object).where(~mask, np.nan)
    for col in ["Study_Hours_Per_Day", "Attendance_Percentage", "Assignment_Scores", "Sleep_Hours"]:
        df[col] = pd.to_numeric(df[col])

    # --- Add a few exact duplicate records, then shuffle ---
    dupes = df.sample(N_DUPLICATES, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main():
    df = generate()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print("Synthetic dataset created.")
    print(f"Saved to : {DATA_PATH}")
    print(f"Shape    : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing  : {int(df.isna().sum().sum())} cells")
    print(f"Duplicates: {int(df.duplicated().sum())} rows")
    print("\nFirst 10 records:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
