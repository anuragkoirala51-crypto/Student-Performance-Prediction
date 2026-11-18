# Student Project and Presentation Guide
### Machine Learning College Coursework Project
**Student Name:** Anurag Koirala  
**Roll Number:** ECB24030  
**Project:** Student Academic Performance Prediction System  
**Dataset:** UCI Student Performance Benchmark (Cortez, 2008)  

---

## 1. The Core Idea in 30 Seconds

> Suppose a college or university department has **1,000 students**, but faculty mentors only have enough hours to offer intensive 1-on-1 remedial tutoring to **200 students** (20% of the batch).
> 
> **The academic challenge:** Which 200 students should receive that extra support?
> 
> Students who struggle academically rarely do so without early indicators. At the **start of the semester**, key factors are already visible: past failed subjects/backlogs, low weekly self-study hours, and lack of tutoring or family support.
> 
> Our machine learning system evaluates student study habits and academic history to identify students at risk of failing early, so professors and mentors can step in before final semester exams.

---

## 2. The Flaw in Typical Student Projects

Most standard student projects analyzing the Cortez (2008) dataset make a critical methodological error:
* They include **Term 1 ($G1$)** and **Term 2 ($G2$)** exam grades as predictors for the **Final Grade ($G3$)**.
* Because Term 2 takes place only two weeks before the final exam, the correlation between $G2$ and $G3$ is **$0.90$**.
* The model easily reports **90%+ accuracy**, but this is not an early warning—it is simply reading student report cards at the end of the year when it is too late to intervene.

### Our Solution: The 4 Chronological Setups

| Setup | Practical Name | Timing | Inputs Available |
|---|---|---|---|
| **Setup A** | **End-of-Year Diagnostic** | Late in the year | Full data ($G1, G2$, attendance). High accuracy (87%), but too late for early warning. |
| **Setup B** | **Mid-Year Check** | Middle of the year | Term 1 grade ($G1$) available, but no $G2$. Useful for mid-term adjustments. |
| **Setup C** | **Conventional Early Warning** | Early in the year | Drops prior grades, but retains year-end attendance data. |
| **Setup D** | **Day-1 Enrolment Model** | First day of school | **No grades, no attendance.** Evaluates strictly pre-matriculation habits and history. |

---

## 3. Key Technical Contributions Explained Simply

### 1. Day-1 Enrolment Model (Setup D)
* **Finding:** On Day 1, predicting an exact continuous grade (e.g., $11.4$ vs $12.2$ out of 20) has an $R^2 \approx 0.03$. Exact point prediction is mathematically unreliable before coursework begins.
* **Solution:** While exact grade regression fails, **risk classification remains effective (ROC-AUC = 0.635)**. The model reliably identifies who is at risk versus who is on track.

### 2. Teacher-Capacity Evaluation (Recall@K)
* Instead of reporting abstract accuracy, we measure practical institutional utility:
* **Result:** By screening only the top **20% most at-risk students**, counsellors catch **30.8% of all failing students**. This is **1.52 times more effective** than random student selection.

### 3. Cost-Sensitive Decision Cutoff
* A standard machine learning model uses an arbitrary 50% probability cutoff.
* In education, failing to identify an at-risk student (False Negative) is far worse than offering unnecessary tutoring to a student who passes (False Positive).
* By weighting False Negatives 5 times heavier than False Positives, we calculate the optimal decision threshold at **28% risk** ($\theta^* = 0.28$), reducing institutional policy loss by **49.8%**.

### 4. Actionable Study Time Impact Analysis
* The dashboard avoids fatalistic demographic profiling (such as parental education or home address).
* Instead, it provides actionable recourse: showing how specific increases in weekly study hours (e.g., moving from $<2$ hours to $2-5$ hours) reduce estimated failure risk.

### 5. The Zero-Grade Anomaly
* In the dataset, 38 students scored exactly 0 on the final exam.
* In Portuguese schools, a grade of 0 represents non-attendance or administrative withdrawal rather than a typical test score. Recognizing this prevents distorted linear regression interpretations.

---

## 4. Two-Minute Presentation Script (By Anurag Koirala - ECB24030)

> "Good morning. My name is Anurag Koirala (Roll No: ECB24030), and my college coursework project is a Machine Learning System for Student Academic Performance Prediction.
> 
> Many standard student projects report 90%+ accuracy by simply feeding Term 2 exam marks into the model. However, because Term 2 takes place just before finals, that isn't true early prediction—it's just reading report cards when it's already too late to intervene.
> 
> To solve this, I built a **Semester Start model (Setup D)**. Before any exams have taken place, the system evaluates baseline study habits, past failed subjects/backlogs, and academic goals.
> 
> Under realistic mentoring capacity, by screening only the top 20% most at-risk students, our model successfully catches **almost a third (30.8%) of all students who would otherwise fail**—a 1.5x efficiency lift over random triage.
> 
> Finally, the interactive Gradio dashboard allows students and professors to test profiles in real time and see exactly how increasing self-study hours directly improves passing probability."

---

## 5. Defense Cheat Sheet: Common Examiner Questions

* **Question: Why did you not use deep neural networks?**
  * **Answer:** With 395 samples in tabular format, deep neural networks are prone to severe overfitting. Ensembles of decision trees (Random Forest and Gradient Boosting) offer superior tabular performance and interpretable feature attributions.

* **Question: Why did your $R^2$ drop to near-zero in Setup D?**
  * **Answer:** Because on Day 1, no academic performance data exists. Predicting exact numerical grades without examination history is not statistically feasible. However, risk classification remains effective ($\text{ROC-AUC} = 0.635$), which is what schools require for triage.

* **Question: How did you ensure the model is fair to different demographic groups?**
  * **Answer:** I conducted a subgroup fairness audit evaluating equalized odds across gender and rural/urban geography. In addition, the interactive simulator isolates actionable habits (such as study hours and tutoring) from immutable demographics.
