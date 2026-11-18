"""
Early Warning System for Student Performance: Training Pipeline
--------------------------------------------------------------
End-to-end pipeline covering Setups A, B, C, D across Regression and Classification.
Includes:
- Feature timeline audit (Day-0 Setup D: drops G1, G2, absences)
- Zero-grade dropout anomaly handling & analysis
- Cross-setup degradation comparison (A -> B -> C -> D)
- Recall@K (Teacher-capacity curve & lift over random review)
- Calibration analysis (Brier score & reliability curves)
- Cost-sensitive threshold optimization (CFN=5x, CFP=1x)
- Subgroup fairness audit (Sex & Urban/Rural demographic parity)
- Explainable AI with SHAP (Actionable vs Immutable factors)
- Model serialization for the live interactive demo
"""

import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)

# Optional XGBoost import with safe fallback for systems without OpenMP
try:
    import xgboost as xgb
    # Check if libxgboost dynamic library actually works without throwing OpenMP errors
    _dummy = xgb.XGBClassifier(n_estimators=1)
    HAS_XGB = True
    print("XGBoost available and active.")
except Exception as e:
    HAS_XGB = False
    print(f"XGBoost unavailable ({type(e).__name__}); using GradientBoosting instead.")

# SHAP check
try:
    import shap
    HAS_SHAP = True
    print("SHAP available and active.")
except Exception as e:
    HAS_SHAP = False
    print(f"SHAP unavailable ({e}).")

warnings.filterwarnings("ignore")

# ----------------- Reproducibility & Constants -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

PASS_MARK = 10
TEST_SIZE = 0.20
CV_FOLDS = 5
FIG_DIR = "figures"
MODEL_DIR = "models"
DATA_DIR = "data"

for d in [FIG_DIR, MODEL_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

def save_fig(name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path)
    plt.close()
    print(f"  [Saved Figure] {path}")

# ----------------- Data Loading -----------------
def load_data():
    """Load the Mathematics cohort from local directory or download directly."""
    csv_path = os.path.join(DATA_DIR, "student-mat.csv")
    if not os.path.exists(csv_path):
        import urllib.request, zipfile, io
        url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
        print(f"Downloading dataset archive from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            outer_zip = zipfile.ZipFile(io.BytesIO(resp.read()))
            outer_zip.extractall(DATA_DIR)
        
        inner_zip_path = os.path.join(DATA_DIR, "student.zip")
        if os.path.exists(inner_zip_path):
            with zipfile.ZipFile(inner_zip_path) as inner_zip:
                inner_zip.extractall(DATA_DIR)

    df = pd.read_csv(csv_path, sep=";") if ";" in open(csv_path).readline() else pd.read_csv(csv_path)
    df["pass"] = (df["G3"] >= PASS_MARK).astype(int)
    print(f"Dataset successfully loaded: {df.shape[0]} students, {df.shape[1]} columns.")
    return df

# ----------------- Feature Partitioning -----------------
SETUPS = {
    "Setup A (Full/All Features)": [],
    "Setup B (Mid-Term: No G2)": ["G2"],
    "Setup C (No Prior Grades)": ["G1", "G2"],
    "Setup D (Day-0: True Enrolment)": ["G1", "G2", "absences"]
}

def build_preprocessor(X):
    """Constructs leak-free ColumnTransformer for numerical and categorical features."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor

# ----------------- EDA Plots -----------------
def run_eda(df):
    print("\n--- Generating Comprehensive EDA Figures ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. G3 Distribution with Zero-Grade Dropout Anomaly
    sns.histplot(df["G3"], bins=21, kde=True, ax=axes[0, 0], color="#1f77b4", discrete=True)
    axes[0, 0].axvline(PASS_MARK, color="red", linestyle="--", linewidth=1.5, label=f"Pass Threshold ({PASS_MARK})")
    zero_count = (df["G3"] == 0).sum()
    axes[0, 0].annotate(f"Zero-Grade Spike\n({zero_count} dropouts/absences)", xy=(0, zero_count), xytext=(2.5, 45),
                        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6))
    axes[0, 0].set_title("1. Final Grade (G3) & Zero-Grade Anomaly")
    axes[0, 0].set_xlabel("Final Grade (0-20 scale)")
    axes[0, 0].set_ylabel("Student Count")
    axes[0, 0].legend()

    # 2. Key Predictor Correlation
    corr_cols = [c for c in ["G1", "G2", "G3", "absences", "studytime", "failures", "age"] if c in df.columns]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="mako", vmin=-1, vmax=1, ax=axes[0, 1])
    axes[0, 1].set_title("2. Predictor Correlation Matrix (G1/G2 Dominance)")

    # 3. Fail Rate by Past Failures (Day-0 Indicator)
    failure_rates = df.groupby("failures")["pass"].apply(lambda s: (1 - s.mean()) * 100).reset_index()
    sns.barplot(data=failure_rates, x="failures", y="pass", ax=axes[1, 0], palette="Reds_d")
    axes[1, 0].set_title("3. Failure Rate (%) by Prior Academic Failures")
    axes[1, 0].set_xlabel("Past Academic Failures")
    axes[1, 0].set_ylabel("Actual Failure Rate (%)")

    # 4. Actionable Levers: Study Time & Absences
    palette = {0: "#e74c3c", 1: "#2ecc71"}
    sns.boxplot(data=df, x="studytime", y="absences", hue="pass", palette=palette, ax=axes[1, 1], showfliers=False)
    axes[1, 1].set_title("4. Actionable Levers: Absences by Study Time Tier")
    axes[1, 1].set_xlabel("Weekly Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
    axes[1, 1].set_ylabel("Absences (days)")
    handles, _ = axes[1, 1].get_legend_handles_labels()
    axes[1, 1].legend(handles, ["Fails (<10)", "Passes (>=10)"], title="Outcome")

    plt.tight_layout()
    save_fig("eda_comprehensive_overview")

# ----------------- Model Training & Benchmark -----------------
def evaluate_setups(df):
    print("\n--- Training & Evaluating Models Across Setups A, B, C, D ---")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["pass"])
    print(f"Training set: {len(train_df)} | Test set: {len(test_df)}")

    reg_results = []
    clf_results = []
    trained_pipelines = {}

    for setup_name, drop_cols in SETUPS.items():
        print(f"  Training {setup_name}...")
        X_train = train_df.drop(columns=["G3", "pass"] + drop_cols)
        y_train_reg = train_df["G3"]
        y_train_clf = train_df["pass"]

        X_test = test_df.drop(columns=["G3", "pass"] + drop_cols)
        y_test_reg = test_df["G3"]
        y_test_clf = test_df["pass"]

        preprocessor = build_preprocessor(X_train)

        # Regression Models
        reg_models = {
            "Baseline (Mean)": DummyRegressor(strategy="mean"),
            "Ridge Regression": Ridge(alpha=1.0, random_state=SEED),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=SEED),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=SEED)
        }
        if HAS_XGB:
            reg_models["XGBoost Regressor"] = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=SEED, eval_metric="rmse")

        for m_name, model in reg_models.items():
            pipe = Pipeline([("prep", preprocessor), ("reg", model)])
            pipe.fit(X_train, y_train_reg)
            preds = pipe.predict(X_test)

            mae = mean_absolute_error(y_test_reg, preds)
            rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
            r2 = r2_score(y_test_reg, preds)

            reg_results.append({
                "Setup": setup_name,
                "Model": m_name,
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "R2": round(r2, 3)
            })
            trained_pipelines[f"reg_{setup_name}_{m_name}"] = pipe

        # Classification Models
        clf_models = {
            "Baseline (Majority)": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=150, max_depth=5, class_weight="balanced", random_state=SEED),
            "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=SEED)
        }
        if HAS_XGB:
            clf_models["XGBoost Classifier"] = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=SEED, eval_metric="logloss")

        for m_name, model in clf_models.items():
            pipe = Pipeline([("prep", preprocessor), ("clf", model)])
            pipe.fit(X_train, y_train_clf)
            preds = pipe.predict(X_test)

            if hasattr(pipe, "predict_proba"):
                probs_pass = pipe.predict_proba(X_test)[:, 1]
                probs_fail = 1.0 - probs_pass
                roc_auc = roc_auc_score(y_test_clf, probs_pass)
                pr_auc = average_precision_score(1 - y_test_clf, probs_fail)
                brier = brier_score_loss(1 - y_test_clf, probs_fail)
            else:
                probs_pass = np.zeros(len(y_test_clf))
                probs_fail = np.zeros(len(y_test_clf))
                roc_auc = 0.5
                pr_auc = 0.0
                brier = 0.0

            acc = accuracy_score(y_test_clf, preds)
            rec_fail = recall_score(y_test_clf, preds, pos_label=0, zero_division=0)
            prec_fail = precision_score(y_test_clf, preds, pos_label=0, zero_division=0)
            f1_fail = f1_score(y_test_clf, preds, pos_label=0, zero_division=0)

            clf_results.append({
                "Setup": setup_name,
                "Model": m_name,
                "Accuracy": round(acc, 3),
                "Fail_Recall": round(rec_fail, 3),
                "Fail_Precision": round(prec_fail, 3),
                "Fail_F1": round(f1_fail, 3),
                "ROC_AUC": round(roc_auc, 3),
                "PR_AUC_Fail": round(pr_auc, 3),
                "Brier_Score": round(brier, 3)
            })
            trained_pipelines[f"clf_{setup_name}_{m_name}"] = pipe

    df_reg = pd.DataFrame(reg_results)
    df_clf = pd.DataFrame(clf_results)

    df_reg.to_csv("benchmark_regression.csv", index=False)
    df_clf.to_csv("benchmark_classification.csv", index=False)
    return train_df, test_df, df_reg, df_clf, trained_pipelines

# ----------------- Plot Degradation -----------------
def plot_setup_degradation(df_reg, df_clf):
    print("\n--- Plotting Setup Degradation (A -> B -> C -> D) ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Regression
    best_reg = df_reg[df_reg["Model"] != "Baseline (Mean)"].groupby("Setup")["R2"].max().reset_index()
    setup_labels = {
        "Setup A (Full/All Features)": "Setup A (Full)",
        "Setup B (Mid-Term: No G2)": "Setup B (No G2)",
        "Setup C (No Prior Grades)": "Setup C (No Grades)",
        "Setup D (Day-0: True Enrolment)": "Setup D (Day-0)"
    }
    best_reg["Setup_Short"] = best_reg["Setup"].map(setup_labels)
    best_reg = best_reg.sort_values(by="R2", ascending=False)
    sns.barplot(data=best_reg, x="Setup_Short", y="R2", palette="crest", ax=axes[0])
    axes[0].set_title("Regression R² Score Drop (True Early Warning Penalty)")
    axes[0].set_ylabel("R² Score (Variance Explained)")
    axes[0].set_ylim(-0.1, 1.0)
    for p in axes[0].patches:
        val = p.get_height()
        axes[0].annotate(f"{val:.2f}", (p.get_x() + p.get_width() / 2., val),
                         ha="center", va="center", xytext=(0, 8), textcoords="offset points", fontweight="bold")

    # Classification
    best_clf = df_clf[df_clf["Model"] != "Baseline (Majority)"].groupby("Setup")[["ROC_AUC", "Fail_Recall"]].max().reset_index()
    best_clf["Setup_Short"] = best_clf["Setup"].map(setup_labels)
    melted = best_clf.melt(id_vars="Setup_Short", value_vars=["ROC_AUC", "Fail_Recall"], var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="Setup_Short", y="Score", hue="Metric", palette="Blues_r", ax=axes[1])
    axes[1].set_title("Classification Metric Progression across Setups")
    axes[1].set_ylabel("Score (0.0 to 1.0)")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(title="Metric", loc="lower left")

    plt.tight_layout()
    save_fig("setup_degradation_comparison")

# ----------------- Differentiator 1: Recall@K -----------------
def evaluate_recall_at_k(pipe, X, y_pass):
    print("\n--- Evaluating Recall@K (Teacher Capacity Curve) ---")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    p_pass = cross_val_predict(pipe, X, y_pass, cv=skf, method="predict_proba")[:, 1]
    p_fail = 1.0 - p_pass
    y_fail = (1 - np.asarray(y_pass)).astype(int)

    order = np.argsort(-p_fail)
    n, total_fails = len(y_fail), y_fail.sum()

    ks = [0.10, 0.20, 0.30, 0.40, 0.50]
    rows = []
    for k in ks:
        m = int(np.ceil(k * n))
        caught = y_fail[order[:m]].sum()
        recall = caught / total_fails
        precision = caught / m
        lift = precision / (total_fails / n)
        rows.append({
            "Capacity (Top K%)": f"{int(k*100)}%",
            "Students Reviewed": m,
            "Failing Caught": int(caught),
            "Total Failing": total_fails,
            "Fail Recall (%)": f"{recall*100:.1f}%",
            "Precision (%)": f"{precision*100:.1f}%",
            "Lift vs Random": f"{lift:.2f}x"
        })
    df_k = pd.DataFrame(rows)
    print(df_k.to_string(index=False))
    df_k.to_csv("teacher_capacity_table.csv", index=False)

    cum_fails = np.cumsum(y_fail[order]) / total_fails
    x_axis = np.arange(1, n + 1) / n

    plt.figure(figsize=(7, 5))
    plt.plot(x_axis * 100, cum_fails * 100, color="#d62728", lw=2.5, label="Model Early Warning Curve")
    plt.plot([0, 100], [0, 100], "--", color="gray", label="Random Student Review (No Model)")
    plt.axvline(20, color="green", linestyle=":", label="20% Counsellor Capacity Line")
    
    idx_20 = int(np.ceil(0.20 * n)) - 1
    recall_20 = cum_fails[idx_20] * 100
    plt.plot(20, recall_20, "ro", markersize=8)
    plt.annotate(f"Top 20% review catches {recall_20:.1f}% of all fails",
                 xy=(20, recall_20), xytext=(28, recall_20 - 8),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
                 fontweight="bold")

    plt.xlabel("% of Students Screened (Ranked by Model Risk)", fontsize=11)
    plt.ylabel("% of Failing Students Detected (Recall)", fontsize=11)
    plt.title("Teacher-Capacity Fail Capture Curve (Setup D Day-0)", fontsize=12, fontweight="bold")
    plt.xlim(0, 100)
    plt.ylim(0, 105)
    plt.legend(loc="lower right")
    save_fig("teacher_capacity_curve")
    return df_k

# ----------------- Differentiator 2: Calibration -----------------
def evaluate_calibration(pipe, X, y_pass):
    print("\n--- Evaluating Calibration & Reliability Curve ---")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    p_pass = cross_val_predict(pipe, X, y_pass, cv=skf, method="predict_proba")[:, 1]
    p_fail = 1.0 - p_pass
    y_fail = (1 - np.asarray(y_pass)).astype(int)

    calibrated_pipe = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=3)
    calibrated_pipe.fit(X, y_pass)
    p_pass_cal = calibrated_pipe.predict_proba(X)[:, 1]
    p_fail_cal = 1.0 - p_pass_cal

    brier_raw = brier_score_loss(y_fail, p_fail)
    brier_cal = brier_score_loss(y_fail, p_fail_cal)
    print(f"Raw Model Brier Score        : {brier_raw:.4f}")
    print(f"Calibrated Model Brier Score : {brier_cal:.4f}")

    frac_raw, mean_raw = calibration_curve(y_fail, p_fail, n_bins=5, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_fail, p_fail_cal, n_bins=5, strategy="quantile")

    plt.figure(figsize=(6, 5))
    plt.plot(mean_raw, frac_raw, "s-", color="#1f77b4", label=f"Uncalibrated (Brier={brier_raw:.3f})")
    plt.plot(mean_cal, frac_cal, "o-", color="#2ca02c", label=f"Calibrated Sigmoid (Brier={brier_cal:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Reliability")
    plt.xlabel("Mean Predicted Probability of Failure")
    plt.ylabel("Observed Proportion of Failures")
    plt.title("Calibration Reliability Curve (Setup D Day-0)")
    plt.legend(loc="upper left")
    save_fig("calibration_reliability_curve")

# ----------------- Differentiator 3: Cost-Sensitive Threshold -----------------
def optimize_decision_threshold(pipe, X, y_pass, cost_fn=5.0, cost_fp=1.0):
    print("\n--- Cost-Sensitive Threshold Optimization ---")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    p_pass = cross_val_predict(pipe, X, y_pass, cv=skf, method="predict_proba")[:, 1]
    p_fail = 1.0 - p_pass
    y_fail = (1 - np.asarray(y_pass)).astype(int)

    thresholds = np.linspace(0.10, 0.90, 81)
    costs = []

    for t in thresholds:
        pred_fail = (p_fail >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_fail, pred_fail).ravel()
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(total_cost)

    best_idx = np.argmin(costs)
    opt_t = thresholds[best_idx]
    opt_cost = costs[best_idx]
    default_idx = np.abs(thresholds - 0.50).argmin()
    default_cost = costs[default_idx]

    print(f"Default Cutoff (0.50) Loss : {default_cost:.1f}")
    print(f"Optimal Cutoff (theta*)   : {opt_t:.2f} | Loss: {opt_cost:.1f} (Cost Reduction: {(1 - opt_cost/default_cost)*100:.1f}%)")

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, costs, color="#e6550d", lw=2.5, label="Total Educational Loss Curve")
    plt.axvline(0.50, color="gray", linestyle="--", label=f"Default 0.50 Cutoff (Loss={default_cost:.0f})")
    plt.axvline(opt_t, color="green", linestyle="-.", label=f"Optimal Cutoff θ*={opt_t:.2f} (Loss={opt_cost:.0f})")
    plt.scatter([opt_t], [opt_cost], color="green", s=80, zorder=5)
    plt.xlabel("Fail Probability Cutoff Threshold (θ)")
    plt.ylabel(f"Total Policy Cost (Loss = {cost_fn:.0f}×FN + {cost_fp:.0f}×FP)")
    plt.title("Cost-Sensitive Threshold Optimization Curve")
    plt.legend()
    save_fig("cost_sensitive_threshold_optimization")
    return opt_t

# ----------------- Differentiator 4: Fairness Audit -----------------
def evaluate_subgroup_fairness(pipe, df_test, opt_threshold=0.35):
    print("\n--- Subgroup Fairness & Disparate Impact Audit ---")
    X_test_d = df_test.drop(columns=["G3", "pass"] + SETUPS["Setup D (Day-0: True Enrolment)"])
    p_pass = pipe.predict_proba(X_test_d)[:, 1]
    p_fail = 1.0 - p_pass
    pred_fail = (p_fail >= opt_threshold).astype(int)
    y_fail = (1 - df_test["pass"]).astype(int)

    test_eval = df_test.copy()
    test_eval["pred_fail"] = pred_fail
    test_eval["actual_fail"] = y_fail

    fairness_rows = []
    for sensitive_attr in ["sex", "address"]:
        for grp in test_eval[sensitive_attr].unique():
            sub = test_eval[test_eval[sensitive_attr] == grp]
            flag_rate = sub["pred_fail"].mean()
            actual_rate = sub["actual_fail"].mean()
            
            tp = ((sub["pred_fail"] == 1) & (sub["actual_fail"] == 1)).sum()
            actual_fails = (sub["actual_fail"] == 1).sum()
            tpr = tp / actual_fails if actual_fails > 0 else 0.0

            fp = ((sub["pred_fail"] == 1) & (sub["actual_fail"] == 0)).sum()
            actual_passes = (sub["actual_fail"] == 0).sum()
            fpr = fp / actual_passes if actual_passes > 0 else 0.0

            fairness_rows.append({
                "Attribute": sensitive_attr,
                "Group": grp,
                "Count": len(sub),
                "Base Fail Rate": f"{actual_rate*100:.1f}%",
                "Model Flag Rate": f"{flag_rate*100:.1f}%",
                "Fail Recall (TPR)": f"{tpr*100:.1f}%",
                "False Alarm Rate (FPR)": f"{fpr*100:.1f}%"
            })

    fairness_df = pd.DataFrame(fairness_rows)
    print(fairness_df.to_string(index=False))
    fairness_df.to_csv("fairness_audit.csv", index=False)
    return fairness_df

# ----------------- Differentiator 5: SHAP & Actionable Recourse -----------------
def run_shap_analysis(pipe, X_train, X_test):
    if not HAS_SHAP:
        print("\nSHAP library not available, skipping SHAP plots.")
        return

    print("\n--- Computing SHAP Values for Setup D (Day-0 Model) ---")
    preprocessor = pipe.named_steps["prep"]
    model = pipe.named_steps["clf"]

    # Transform features
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    # TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)
    # Binary classification SHAP values: focus on Fail class (class index 0)
    shap_vals = explainer.shap_values(X_test_trans)
    
    # Random Forest returns list [n_classes] of shape (n_samples, n_features) or 3D array
    if isinstance(shap_vals, list):
        shap_fail = shap_vals[0] # class 0 = Fail
    elif len(shap_vals.shape) == 3:
        shap_fail = shap_vals[:, :, 0]
    else:
        shap_fail = shap_vals

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_fail, X_test_trans, feature_names=feature_names, show=False, max_display=12)
    plt.title("SHAP Summary: Day-0 Predictors of Student Failure (Setup D)", fontsize=12)
    save_fig("shap_summary_setup_d")

# ----------------- Main Execution -----------------
def main():
    print("=" * 70)
    print("EARLY WARNING STUDENT PERFORMANCE: FULL END-TO-END PIPELINE")
    print("=" * 70)

    # 1. Load Data
    df = load_data()

    # 2. Run EDA
    run_eda(df)

    # 3. Model Benchmark across Setups
    train_df, test_df, df_reg, df_clf, pipelines = evaluate_setups(df)

    # 4. Plot Degradation (A -> B -> C -> D)
    plot_setup_degradation(df_reg, df_clf)

    # 5. Extract Best Setup D Model (Day-0 Model)
    best_pipe_d = pipelines["clf_Setup D (Day-0: True Enrolment)_Random Forest Classifier"]
    X_train_d = train_df.drop(columns=["G3", "pass"] + SETUPS["Setup D (Day-0: True Enrolment)"])
    X_test_d = test_df.drop(columns=["G3", "pass"] + SETUPS["Setup D (Day-0: True Enrolment)"])
    y_train_pass = train_df["pass"]

    # 6. Run Differentiators
    evaluate_recall_at_k(best_pipe_d, X_train_d, y_train_pass)
    evaluate_calibration(best_pipe_d, X_train_d, y_train_pass)
    opt_t = optimize_decision_threshold(best_pipe_d, X_train_d, y_train_pass, cost_fn=5.0, cost_fp=1.0)
    evaluate_subgroup_fairness(best_pipe_d, test_df, opt_threshold=opt_t)
    run_shap_analysis(best_pipe_d, X_train_d, X_test_d)

    # 7. Model Serialization
    print("\n--- Serializing Models for Web Demo & Deployment ---")
    joblib.dump(best_pipe_d, os.path.join(MODEL_DIR, "best_classifier_setup_d.joblib"))
    
    pipe_a = pipelines["clf_Setup A (Full/All Features)_Random Forest Classifier"]
    joblib.dump(pipe_a, os.path.join(MODEL_DIR, "best_classifier_setup_a.joblib"))

    best_reg_d = pipelines["reg_Setup D (Day-0: True Enrolment)_Random Forest Regressor"]
    joblib.dump(best_reg_d, os.path.join(MODEL_DIR, "best_regressor_setup_d.joblib"))

    # Save a template student row for demo testing
    sample_student = test_df.iloc[[0]].drop(columns=["G3", "pass"] + SETUPS["Setup D (Day-0: True Enrolment)"])
    sample_student.to_csv("data/sample_student.csv", index=False)
    print("Saved sample student to 'data/sample_student.csv'")

    print("\n======================================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY! ALL ARTIFACTS READY.")
    print("======================================================================")

if __name__ == "__main__":
    main()
