# Student Performance Prediction System Using Random Forest Classifier

An AIML mini-project that predicts whether a student's academic performance is **Good**, **Average** or **Poor**
from study habits and marks, using a **Random Forest Classifier** (Scikit-learn) and an interactive **Streamlit** dashboard.

> The dataset is **synthetically generated** (no real student records). It is stored as a structured CSV file, not a database.

## Folder structure
```
student_performance_prediction/
├── dataset/student_performance.csv     # generated dataset (1,840 raw rows)
├── models/random_forest_model.pkl      # saved preprocessing + Random Forest pipeline
├── models/metrics.json                 # real evaluation results written by train_model.py
├── notebooks/analysis.ipynb            # step-by-step analysis notebook
├── reports/                            # EDA graphs, confusion matrix, text reports (auto-created)
├── common.py                           # shared constants, validation, prediction helpers
├── generate_dataset.py                 # creates the synthetic dataset
├── train_model.py                      # preprocessing, EDA, training, evaluation, saving
├── app.py                              # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Requirements
Python **3.9 or newer** (3.10 - 3.12 recommended). Check with `python --version` (Windows) or `python3 --version` (macOS).

## Setup and run (VS Code)

1. Create/unzip the project folder, then **File > Open Folder** in VS Code and open a terminal (**Terminal > New Terminal**).

2. Create a virtual environment

   | | macOS | Windows (PowerShell) |
   |---|---|---|
   | create | `python3 -m venv venv` | `python -m venv venv` |
   | activate | `source venv/bin/activate` | `venv\Scripts\Activate.ps1` |

   Windows tip: if activation is blocked, run `Set-ExecutionPolicy -Scope Process RemoteSigned` first (or use Command Prompt: `venv\Scripts\activate.bat`).

3. Install dependencies: `pip install -r requirements.txt`

4. Generate the dataset: `python generate_dataset.py` (macOS: `python3 ...` if needed; inside the venv `python` works)

5. Train and evaluate the model: `python train_model.py`
   (prints the analysis and metrics, saves graphs in `reports/` and the model in `models/`)

6. Start the dashboard: `streamlit run app.py`

7. The browser opens automatically. If not, go to **http://localhost:8501**. Stop the app with `Ctrl + C`.

Optional notebook: `jupyter notebook` and open `notebooks/analysis.ipynb` (or use the VS Code Jupyter extension, select the `venv` kernel).

## Results (from the included run)
Measured on the 20 % unseen test set (360 students), `random_state=42`:
accuracy 75.0 %, macro precision 75.8 %, macro recall 75.5 %, macro F1 75.6 %;
5-fold CV accuracy 70.6 % ± 1.2 %. Re-running `train_model.py` reproduces these numbers exactly.
They are intentionally not near 100 %, because the categories overlap like real student data.

## How data leakage is avoided
* Duplicates are removed **before** splitting, so identical rows cannot land in both train and test sets.
* Missing values are filled and categories encoded **inside a scikit-learn Pipeline** that is fitted on training data only.
* The same saved pipeline is used by the dashboard, so prediction uses exactly the training preprocessing.

## Troubleshooting
* `Dataset not found` / `Model not found` - run steps 4 and 5 first.
* `ModuleNotFoundError` - the virtual environment is not activated, or step 3 was skipped.
* Version warning when loading the model - delete `models/random_forest_model.pkl` and rerun `python train_model.py` on the same machine.
