"""
Generates Early_Warning_Student_Performance.ipynb programmatically.
Easier to maintain this way than editing the raw JSON by hand.
"""

import json

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })


add_md("""# Student Performance Early Warning System
### Machine Learning & Deep Learning Assignment Project
**Student Name:** Anurag Koirala  
**Roll Number:** ECB24030  
**Dataset:** UCI Student Performance Benchmark (Cortez, 2008), doi:10.24432/C5TG7T  
**Tasks:** Continuous Grade Regression ($G3: 0-20$) and Cost-Sensitive Classification (Pass vs. Fail)  
**Core Innovation:** Comparing **Day-1 Enrolment Screening** (no tests taken yet) vs. **Late-Stage Diagnostics** (with test leakage)

---
## 1a. Quick Setup & Package Installs""")


add_code("""# install required packages — only needed first run on Colab
!pip install -q ucimlrepo shap gradio imbalanced-learn joblib""")


add_code("""import os, sys, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

PASS_MARK = 10          # 10/20 is passing
TEST_SIZE = 0.20
CV_FOLDS = 5
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

def save_fig(name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path)
    print("Saved:", path)

print("Anurag Koirala (ECB24030) | Seed:", SEED, "| Pass mark:", PASS_MARK)""")


add_md("""---
## 1b. The Information Leakage Flaw & Dataset Loading

### The Flaw in Typical Student Projects
Most projects using the Cortez (2008) dataset report 90%+ accuracy by including Term 1 ($G1$) and Term 2 ($G2$) grades.
However, Term 2 is recorded weeks before final exams with a **0.90 correlation** to $G3$. That's not early warning — that's late diagnosis.

### Our 4 Experimental Setups
1. **Setup A (End-of-Year):** Full features ($G1, G2$, attendance).
2. **Setup B (Mid-Year):** First-term snapshot (keeps $G1$, drops $G2$).
3. **Setup C (Conventional):** Drops prior grades, but retains cumulative absences.
4. **Setup D (Day-1 Enrolment):** **Zero tests, zero attendance.** True day-one pre-matriculation triage.""")


add_code("""# try ucimlrepo first, fall back to manual download if it doesn't work
try:
    from ucimlrepo import fetch_ucirepo
    repo = fetch_ucirepo(id=320)
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    print("Fetched via ucimlrepo!")
except Exception as e:
    import urllib.request, zipfile, io
    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        z.extractall("data_temp")
    with zipfile.ZipFile("data_temp/student.zip") as z2:
        df = pd.read_csv(z2.open("student-mat.csv"), sep=";")
    print("Fallback download worked!")

df["pass"] = (df["G3"] >= PASS_MARK).astype(int)
n_pass, n_fail = df["pass"].sum(), len(df) - df["pass"].sum()
print(f"Total: {len(df)} students | Pass: {n_pass} ({n_pass/len(df)*100:.1f}%) | Fail: {n_fail} ({n_fail/len(df)*100:.1f}%)")""")


add_code("""SETUPS = {
    "Setup A (End-of-Year / Full)": [],
    "Setup B (Mid-Year: After Term 1)": ["G2"],
    "Setup C (No Prior Grades)": ["G1", "G2"],
    "Setup D (Day-1 Enrolment Only)": ["G1", "G2", "absences"]
}

for name, drop_cols in SETUPS.items():
    count = df.shape[1] - 2 - len(drop_cols)
    print(f"  {name:<35} -> {count:>2} features")""")


add_code("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# G3 distribution — that spike at 0 is dropouts / skipped finals
sns.histplot(df["G3"], bins=21, kde=True, ax=axes[0, 0], color="#0284c7", discrete=True)
axes[0, 0].axvline(PASS_MARK, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Pass mark ({PASS_MARK}/20)")
zero_count = (df["G3"] == 0).sum()
axes[0, 0].annotate(f"0-grade spike\\n({zero_count} dropouts)", xy=(0, zero_count), xytext=(2.5, 45),
                    arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6))
axes[0, 0].set_title("Final Grade (G3) & Dropout Spike at 0")
axes[0, 0].legend()

# correlation — G1/G2 dominate, that's the whole point
corr_cols = [c for c in ["G1", "G2", "G3", "absences", "studytime", "failures"] if c in df.columns]
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="Blues", vmin=-1, vmax=1, ax=axes[0, 1])
axes[0, 1].set_title("Predictor Correlations (G1 & G2 leakage visible)")

failure_rates = df.groupby("failures")["pass"].apply(lambda s: (1 - s.mean()) * 100).reset_index()
sns.barplot(data=failure_rates, x="failures", y="pass", ax=axes[1, 0], palette="Reds_d")
axes[1, 0].set_title("Failure Rate (%) by Past Failed Subjects")

palette = {0: "#dc2626", 1: "#059669"}
sns.boxplot(data=df, x="studytime", y="absences", hue="pass", palette=palette, ax=axes[1, 1], showfliers=False)
axes[1, 1].set_title("Actionable Levers: Study Time & Absences by Outcome")

plt.tight_layout()
save_fig("eda_comprehensive_overview")
plt.show()""")


add_code("""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["pass"])

def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="drop"
    )""")


add_code("""from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, recall_score, roc_auc_score, brier_score_loss

reg_results, clf_results, trained_pipelines = [], [], {}

for setup_name, drop_cols in SETUPS.items():
    X_train = train_df.drop(columns=["G3", "pass"] + drop_cols)
    y_train_reg, y_train_clf = train_df["G3"], train_df["pass"]

    X_test = test_df.drop(columns=["G3", "pass"] + drop_cols)
    y_test_reg, y_test_clf = test_df["G3"], test_df["pass"]

    preprocessor = build_preprocessor(X_train)

    # regression
    for m_name, model in [
        ("Baseline (Average)", DummyRegressor(strategy="mean")),
        ("Ridge Regression", Ridge(alpha=1.0, random_state=SEED)),
        ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=SEED)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=SEED))
    ]:
        pipe = Pipeline([("prep", preprocessor), ("reg", model)])
        pipe.fit(X_train, y_train_reg)
        preds = pipe.predict(X_test)
        reg_results.append({
            "Setup": setup_name, "Model": m_name,
            "MAE": round(mean_absolute_error(y_test_reg, preds), 2),
            "R2": round(r2_score(y_test_reg, preds), 2)
        })
        trained_pipelines[f"reg_{setup_name}_{m_name}"] = pipe

    # classification — switched from accuracy to recall/F1 for the fail class, matters more
    for m_name, model in [
        ("Baseline (Majority)", DummyClassifier(strategy="most_frequent")),
        ("Logistic Regression", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
        ("Random Forest", RandomForestClassifier(n_estimators=150, max_depth=5, class_weight="balanced", random_state=SEED)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=SEED))
    ]:
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        pipe.fit(X_train, y_train_clf)
        preds = pipe.predict(X_test)

        if hasattr(pipe, "predict_proba"):
            probs_pass = pipe.predict_proba(X_test)[:, 1]
            probs_fail = 1.0 - probs_pass
            auc = roc_auc_score(y_test_clf, probs_pass)
            brier = brier_score_loss(1 - y_test_clf, probs_fail)
        else:
            probs_pass = probs_fail = np.zeros(len(y_test_clf))
            auc, brier = 0.5, 0.0

        clf_results.append({
            "Setup": setup_name, "Model": m_name,
            "Accuracy": f"{accuracy_score(y_test_clf, preds)*100:.1f}%",
            "Fail Recall": f"{recall_score(y_test_clf, preds, pos_label=0, zero_division=0)*100:.1f}%",
            "ROC-AUC": round(auc, 3),
            "Brier Score": round(brier, 3)
        })
        trained_pipelines[f"clf_{setup_name}_{m_name}"] = pipe

df_reg = pd.DataFrame(reg_results)
df_clf = pd.DataFrame(clf_results)
print("Done training.")""")


add_code("""print("=== REGRESSION RESULTS (Anurag Koirala - ECB24030) ===")
display(df_reg)

print("\\n=== CLASSIFICATION RESULTS ===")
display(df_clf)""")


# teacher capacity eval
add_code("""from sklearn.model_selection import StratifiedKFold, cross_val_predict

best_pipe_d = trained_pipelines["clf_Setup D (Day-1 Enrolment Only)_Random Forest"]
X_train_d = train_df.drop(columns=["G3", "pass"] + SETUPS["Setup D (Day-1 Enrolment Only)"])
y_train_pass = train_df["pass"]

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
p_pass_oof = cross_val_predict(best_pipe_d, X_train_d, y_train_pass, cv=skf, method="predict_proba")[:, 1]
p_fail_oof = 1.0 - p_pass_oof
y_fail_oof = (1 - np.asarray(y_train_pass)).astype(int)

order = np.argsort(-p_fail_oof)
n, total_fails = len(y_fail_oof), y_fail_oof.sum()

rows = []
for k in [0.10, 0.20, 0.30, 0.40, 0.50]:
    m = int(np.ceil(k * n))
    caught = y_fail_oof[order[:m]].sum()
    rows.append({
        "Counsellor Capacity": f"Top {int(k*100)}% riskiest",
        "Students Screened": m,
        "Struggling Students Caught": int(caught),
        "% of All Fails Caught": f"{caught/total_fails*100:.1f}%",
        "Efficiency Lift vs Random": f"{(caught/m)/(total_fails/n):.2f}x"
    })

print("=== TEACHER CAPACITY TRIAGE CURVE (Setup D — Day 1) ===")
display(pd.DataFrame(rows))""")


add_code("""# launch the gradio app inline — make sure app.py is in the same directory
import app
app.demo.launch(inline=True, share=False)""")


notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("Early_Warning_Student_Performance.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook regenerated: Early_Warning_Student_Performance.ipynb")
