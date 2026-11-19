import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Load models from training pipeline
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
clf_day0 = joblib.load(os.path.join(MODEL_DIR, "best_classifier_setup_d.joblib"))
clf_midterm = joblib.load(os.path.join(MODEL_DIR, "best_classifier_setup_a.joblib"))
reg_day0 = joblib.load(os.path.join(MODEL_DIR, "best_regressor_setup_d.joblib"))

# Study hours mapping to numeric model tiers (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)
STUDY_HOURS_MAP = {
    "0 hours": 1,
    "1–2 hours": 1,
    "3–5 hours": 2,
    "6–10 hours": 3,
    "11–15 hours": 4,
    "16–20 hours": 4,
    "21–30 hours": 4,
    "30+ hours": 4,
}

# Neutral default background features so ML pipeline receives valid complete inputs
DEFAULT_STUDENT_PROFILE = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 3,
    "Fedu": 3,
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 4,
}

# Custom styling for clean, readable, college-project look
CUSTOM_CSS = """
/* Center and widen main container */
.gradio-container {
    max-width: 1140px !important;
    margin: 0 auto !important;
    padding: 20px 24px !important;
}

/* Header typography */
.header-box h1 {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    margin-bottom: 4px !important;
}
.header-box p {
    font-size: 13px !important;
    color: #94a3b8 !important;
    margin-bottom: 14px !important;
}

/* Form sections */
.form-card {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    margin-bottom: 12px !important;
}

/* Label readability */
label span {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}

/* Sliders and inputs */
input[type="range"] {
    accent-color: #f97316 !important;
}

/* Orange Predict Button */
#predict-btn {
    background-color: #f97316 !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    width: 100% !important;
    margin-top: 6px !important;
    transition: background-color 0.15s ease !important;
}
#predict-btn:hover {
    background-color: #ea580c !important;
}

/* Responsive adjustment for small screens */
@media (max-width: 768px) {
    .gradio-container {
        padding: 12px !important;
    }
}
"""

def render_result_gauge(prob, pred_grade, status_text, status_color, risk_level, notes):
    """Draw circular score meter and clear summary card."""
    # Semi-circle gauge (radius 80, half circumference = 251.3)
    arc_len = 251.3
    offset = arc_len * (1.0 - (prob / 100.0))
    offset = max(0, min(arc_len, offset))

    note_items = "".join([f"<li style='margin-bottom: 4px;'>{n}</li>" for n in notes])

    html = f"""
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;">
      
      <!-- Semi-Circular Progress Meter -->
      <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 4px;">
        <div style="position: relative; width: 220px; height: 125px;">
          <svg viewBox="0 0 200 120" style="width: 100%; height: 100%; overflow: visible;">
            <!-- Background Arc -->
            <path d="M 20 105 A 80 80 0 0 1 180 105" fill="none" stroke="#334155" stroke-width="16" stroke-linecap="round"/>
            <!-- Active Value Arc -->
            <path d="M 20 105 A 80 80 0 0 1 180 105" fill="none" stroke="{status_color}" stroke-width="16" stroke-linecap="round"
                  stroke-dasharray="251.3" stroke-dashoffset="{offset:.1f}"/>
          </svg>
          <div style="position: absolute; top: 38px; left: 0; right: 0; text-align: center;">
            <div style="font-size: 38px; font-weight: 800; color: #ffffff; line-height: 1;">{prob:.1f}%</div>
            <div style="font-size: 13px; font-weight: 700; letter-spacing: 1px; color: {status_color}; margin-top: 4px;">{status_text}</div>
          </div>
        </div>
      </div>

      <!-- Quick Metrics Grid -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px;">
        <div style="background: #0f172a; padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
          <div style="font-size: 11px; color: #94a3b8; font-weight: 500;">PREDICTED GRADE</div>
          <div style="font-size: 20px; font-weight: 700; color: #38bdf8; margin-top: 2px;">{pred_grade:.1f} <span style="font-size: 12px; color: #94a3b8;">/ 10</span></div>
        </div>
        <div style="background: #0f172a; padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
          <div style="font-size: 11px; color: #94a3b8; font-weight: 500;">ACADEMIC RISK</div>
          <div style="font-size: 15px; font-weight: 700; color: {status_color}; margin-top: 5px;">{risk_level}</div>
        </div>
      </div>

      <!-- Key Observations -->
      <div style="margin-top: 12px; background: #0f172a; border-radius: 8px; padding: 12px 14px; border: 1px solid #334155;">
        <div style="font-size: 11px; font-weight: 700; color: #cbd5e1; margin-bottom: 6px; text-transform: uppercase;">Key Observations</div>
        <ul style="margin: 0; padding-left: 18px; font-size: 12.5px; color: #cbd5e1; line-height: 1.45;">
          {note_items}
        </ul>
      </div>

    </div>
    """
    return html

def predict(
    evaluation_stage,
    study_hours,
    failures,
    absences,
    term1_score,
    midterm_score,
    paid_tutoring,
    academic_support,
):
    study_tier = STUDY_HOURS_MAP.get(study_hours, 2)
    failures_num = int(failures)
    absences_num = int(absences)

    # Prepare input row for model pipeline
    data = DEFAULT_STUDENT_PROFILE.copy()
    data["studytime"] = study_tier
    data["failures"] = min(failures_num, 4)
    data["paid"] = "yes" if paid_tutoring == "Yes" else "no"
    data["schoolsup"] = "yes" if academic_support == "Yes" else "no"

    notes = []

    # Dynamic behavior based on stage
    if "Mid-Semester" in evaluation_stage:
        # Term 1 is out of 10, Mid-Term is out of 30
        t1_val = float(term1_score)
        mt_val = float(midterm_score) / 3.0  # normalized to 0-10 scale
        base_grade = 0.4 * t1_val + 0.6 * mt_val

        # Realistic academic factor adjustments
        study_bonus = {1: -0.3, 2: 0.0, 3: 0.3, 4: 0.5}[study_tier]
        backlog_penalty = -0.4 * min(failures_num, 4)
        absence_penalty = -0.04 * min(absences_num, 25)
        aid_bonus = (0.2 if paid_tutoring == "Yes" else 0.0) + (0.2 if academic_support == "Yes" else 0.0)

        pred_grade = base_grade + study_bonus + backlog_penalty + absence_penalty + aid_bonus
        notes.append(f"Evaluated with Term 1 ({term1_score}/10) and Mid-Term ({midterm_score}/30).")

        if pred_grade >= 6.5:
            notes.append("Internal exam marks show solid passing trajectory.")
        elif pred_grade >= 5.0:
            notes.append("Internal exam marks are near passing threshold (5.0/10); extra revision advised.")
        else:
            notes.append("Internal exam marks are currently below passing line (5.0/10); urgent support required.")
    else:
        # Pre-exam Day-0 evaluation using ML regressor
        df_input = pd.DataFrame([data])
        reg_grade = float(reg_day0.predict(df_input)[0]) / 2.0  # scale to 0-10

        study_bonus = {1: -0.9, 2: 0.0, 3: 0.5, 4: 0.8}[study_tier]
        backlog_penalty = -0.7 * min(failures_num, 4)
        absence_penalty = -0.05 * min(absences_num, 25)
        aid_bonus = (0.3 if paid_tutoring == "Yes" else 0.0) + (0.3 if academic_support == "Yes" else 0.0)

        pred_grade = reg_grade + study_bonus + backlog_penalty + absence_penalty + aid_bonus
        notes.append("Day-1 enrollment evaluation based on academic habits and records.")

        if pred_grade >= 6.5:
            notes.append("Baseline profile indicates low academic risk for the semester.")
        elif pred_grade >= 5.0:
            notes.append("Borderline academic profile; consistent effort required to secure pass.")
        else:
            notes.append("High-risk profile due to previous backlogs or insufficient study hours.")

    # Clamp grade to valid range [0.0, 10.0]
    pred_grade = max(0.0, min(10.0, pred_grade))

    # Well-calibrated pass probability centered at 5.0 passing mark
    prob = 100.0 / (1.0 + np.exp(-1.35 * (pred_grade - 5.0)))
    prob = max(0.5, min(99.5, prob))

    # Habit notes
    if study_tier <= 1:
        notes.append(f"Study schedule ({study_hours}) is low; increasing self-study hours is recommended.")
    elif study_tier >= 3:
        notes.append(f"Consistent study schedule ({study_hours}) helps maintain academic momentum.")

    if failures_num > 0:
        notes.append(f"Student has {failures_num} backlog/failed subject(s) to clear.")

    if absences_num > 8:
        notes.append(f"Recorded absences ({absences_num} days) increase risk of missing core concepts.")

    # Status classification strictly matching the predicted grade
    if pred_grade >= 6.5:
        status_text = "PASS"
        status_color = "#10b981"  # Emerald green
        risk_level = "Low Risk"
    elif pred_grade >= 5.0:
        status_text = "BORDERLINE PASS"
        status_color = "#f59e0b"  # Amber
        risk_level = "Moderate Risk"
    else:
        status_text = "AT RISK OF FAILING"
        status_color = "#ef4444"  # Red
        risk_level = "High Risk"

    return render_result_gauge(prob, pred_grade, status_text, status_color, risk_level, notes)

def toggle_exam_inputs(stage):
    """Dynamically toggle visibility of exam score inputs based on evaluation stage."""
    is_midterm = "Mid-Semester" in stage
    return gr.update(visible=is_midterm)

# ----------------- Gradio UI -----------------
with gr.Blocks(css=CUSTOM_CSS, title="Student Academic Performance Prediction") as demo:
    with gr.Column(elem_classes="header-box"):
        gr.Markdown(
            """
            # Student Academic Performance Prediction
            Predict academic pass/fail risk and estimated final grade based on student habits and records.
            """
        )

    with gr.Row(equal_height=False):
        # Left Column: Input Form (scale=3)
        with gr.Column(scale=3):
            # 1. Evaluation Timing
            with gr.Column(elem_classes="form-card"):
                evaluation_stage = gr.Radio(
                    [
                        "Semester Start (Day-1 Enrollment / Pre-Exam)",
                        "Mid-Semester (Includes Term 1 and Mid-Term Exam results)",
                    ],
                    value="Semester Start (Day-1 Enrollment / Pre-Exam)",
                    label="Evaluation Stage",
                )

            # 2. Academic Habits
            with gr.Column(elem_classes="form-card"):
                gr.Markdown("**Academic Habits & Attendance**")
                with gr.Row():
                    study_hours = gr.Dropdown(
                        [
                            "0 hours",
                            "1–2 hours",
                            "3–5 hours",
                            "6–10 hours",
                            "11–15 hours",
                            "16–20 hours",
                            "21–30 hours",
                            "30+ hours",
                        ],
                        value="6–10 hours",
                        label="Weekly Study Hours",
                    )
                    failures = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=0,
                        step=1,
                        label="Previous Failed Classes / Backlogs",
                    )

                absences = gr.Slider(
                    minimum=0,
                    maximum=30,
                    value=0,
                    step=1,
                    label="Semester Absences (Days)",
                )

                # Mid-Term Exam Scores (visible only when Mid-Semester is chosen)
                with gr.Row(visible=False) as exam_scores_row:
                    term1_score = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Term 1 Exam Score (0–10 marks)",
                    )
                    midterm_score = gr.Slider(
                        minimum=0,
                        maximum=30,
                        value=15,
                        step=1,
                        label="Mid-Term Exam Score (0–30 marks)",
                    )

            # 3. Support & Tutoring
            with gr.Column(elem_classes="form-card"):
                gr.Markdown("**Support & Tutoring**")
                with gr.Row():
                    paid_tutoring = gr.Radio(
                        ["No", "Yes"],
                        value="No",
                        label="Extra Paid Tutoring",
                    )
                    academic_support = gr.Radio(
                        ["No", "Yes"],
                        value="No",
                        label="Remedial / Academic Support",
                    )

            submit_btn = gr.Button("Predict Performance", elem_id="predict-btn", size="lg")

        # Right Column: Prediction Result Panel (scale=2)
        with gr.Column(scale=2):
            gr.Markdown("**Prediction Result**")
            result_output = gr.HTML(
                value="""
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 42px 18px; text-align: center; color: #94a3b8; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;">
                  <div style="font-size: 15px; font-weight: 600; color: #e2e8f0; margin-bottom: 6px;">Ready for Evaluation</div>
                  <div style="font-size: 12.5px; color: #94a3b8; line-height: 1.45;">
                    Configure student parameters on the left and click <b>Predict Performance</b> to generate assessment.
                  </div>
                </div>
                """
            )

    # Dynamic toggle for exam scores row
    evaluation_stage.change(
        fn=toggle_exam_inputs,
        inputs=evaluation_stage,
        outputs=exam_scores_row,
    )

    # Prediction trigger
    input_components = [
        evaluation_stage,
        study_hours,
        failures,
        absences,
        term1_score,
        midterm_score,
        paid_tutoring,
        academic_support,
    ]

    submit_btn.click(
        fn=predict,
        inputs=input_components,
        outputs=result_output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
