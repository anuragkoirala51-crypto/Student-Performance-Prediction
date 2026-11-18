# Student Academic Performance Prediction System
### Machine Learning College Coursework Project

**Author:** Anurag Koirala  
**Roll Number:** ECB24030  
**Course:** Machine Learning / AIML  
**Dataset:** [UCI Student Performance Benchmark](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez, 2008)  
**Live Web Application:** [Access Live Demo](https://qwnik-117-203-160-4.run.pinggy-free.link)  

---

## 📌 Project Overview

Predicting student academic performance is a vital tool for educational institutions to provide timely academic support before final examinations take place.

Most standard machine learning projects on the Cortez (2008) dataset achieve superficial accuracy (>90%) by including **Term 2 ($G2$)** exam grades. Because $G2$ is administered shortly before finals, the correlation between $G2$ and the final grade ($G3$) is **0.90**, rendering the model useless for early intervention.

### Our Solution: Chronological Evaluation Setups
This project evaluates performance across 4 realistic academic intervention points:

1. **Setup D (Semester Start / Day-1):** Strictly evaluates pre-matriculation habits, past failed subjects/backlogs, and study hours without any exam grades or attendance.
2. **Setup C (Conventional Early Warning):** Evaluates baseline attributes and attendance without prior exam marks.
3. **Setup B (Mid-Semester Check):** Includes Term 1 exam marks ($G1$).
4. **Setup A (Diagnostic Full Model):** Uses all features including $G1$ and $G2$.

---

## 🚀 Key Features & Contributions

- **Early Risk Classification:** While predicting exact continuous grades without exam history yields $R^2 \approx 0.03$, risk classification (Pass vs. Fail) achieves strong institutional separation (**ROC-AUC = 0.635** on Day 1).
- **Teacher Capacity Optimization (Recall@K):** Under realistic advising constraints, screening the top **20% most at-risk students captures 30.8% of all eventual failures** (a **1.52x efficiency lift** over random triage).
- **Cost-Sensitive Decision Threshold ($\theta^* = 0.28$):** Because a False Negative (missing a struggling student) is far more detrimental than a False Positive, we optimize the decision threshold to 28%, cutting institutional policy loss by **49.8%**.
- **Interactive Gradio Web Application:** A clean, dark-themed web dashboard for testing student profiles in real time with study-time recourse simulation.

---

## 📁 Repository Structure

```text
├── app.py                                  # Interactive Gradio web application
├── train_pipeline.py                       # Complete model training & evaluation pipeline
├── Early_Warning_Student_Performance.ipynb # Standalone Google Colab-compatible notebook
├── create_notebook.py                      # Programmatic notebook generator
├── STUDENT_GUIDE.md                        # Presentation script and viva defense cheat sheet
├── PROJECT_REPORT.md                       # Comprehensive academic technical report
├── requirements.txt                        # Project dependencies
├── benchmark_classification.csv           # Cross-validation & test metrics for classifiers
├── benchmark_regression.csv               # Regression benchmarks across setups A-D
├── fairness_audit.csv                      # Subgroup fairness audit across gender and location
├── teacher_capacity_table.csv             # Recall@K evaluation table
├── data/                                   # UCI Student Performance raw CSV data
├── figures/                                # Publication-ready visualization charts
│   ├── eda_comprehensive_overview.png
│   ├── setup_degradation_comparison.png
│   ├── teacher_capacity_curve.png
│   ├── calibration_reliability_curve.png
│   ├── cost_sensitive_threshold_optimization.png
│   └── shap_summary_setup_d.png
└── models/                                 # Serialized Scikit-Learn pipelines
    ├── best_classifier_setup_d.joblib
    ├── best_classifier_setup_a.joblib
    └── best_regressor_setup_d.joblib
```

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/anuragkoirala51-crypto/Student-Performance-Prediction.git
cd Student-Performance-Prediction
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Training Pipeline
Trains all regression and classification models across Setups A, B, C, D and exports serialized models and metric tables:
```bash
python train_pipeline.py
```

### 4. Launch the Interactive Web Application
```bash
python app.py
```
Open **`http://localhost:7860`** in your browser to view the application.

---

## 📊 Summary of Results

| Setup | Stage | Classifier | ROC-AUC | Accuracy | Brier Score | Recall@20% |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Setup D** | Semester Start (No exams) | Random Forest | **0.635** | 68.4% | 0.208 | **30.8%** |
| **Setup C** | Early Attendance (No exams)| Random Forest | 0.648 | 69.6% | 0.203 | 34.6% |
| **Setup B** | Mid-Semester ($G1$) | Gradient Boosting| 0.812 | 75.9% | 0.149 | 61.5% |
| **Setup A** | Late Diagnostic ($G1, G2$) | Random Forest | **0.931** | **87.3%** | **0.092** | **88.5%** |

---

## 🎓 Academic Submission

* **Student Name:** Anurag Koirala
* **Roll Number:** ECB24030
* **Coursework:** Machine Learning (AIML) Assignment
