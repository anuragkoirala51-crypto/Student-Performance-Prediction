import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr

# loading the models we saved from train_pipeline
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
clf_day0 = joblib.load(os.path.join(MODEL_DIR, "best_classifier_setup_d.joblib"))
clf_midterm = joblib.load(os.path.join(MODEL_DIR, "best_classifier_setup_a.joblib"))
reg_day0 = joblib.load(os.path.join(MODEL_DIR, "best_regressor_setup_d.joblib"))
reg_midterm = joblib.load(os.path.join(MODEL_DIR, "best_regressor_setup_a.joblib"))

# the model only takes 4 broad tiers so map the dropdown to 1-4
STUDY_HOURS_MAP = {
    "0 hours": 1,
    "1–2 hours": 1,
    "3–5 hours": 2,
    "6–10 hours": 3,
    "11–15 hours": 4,
    "16–20 hours": 4,
    "21–30 hours": 4,
    "31–40 hours": 4,
    "41–55 hours": 4,
    "56–70 hours": 4,
    "70+ hours": 4,
}

# fine-tune offsets because the 4-tier bucketing loses too much resolution
STUDY_FINE_TUNE_MIDTERM = {
    "0 hours": -0.4,
    "1–2 hours": -0.15,
    "3–5 hours": 0.0,
    "6–10 hours": 0.25,
    "11–15 hours": 0.45,
    "16–20 hours": 0.65,
    "21–30 hours": 0.85,
    "31–40 hours": 1.05,
    "41–55 hours": 1.25,
    "56–70 hours": 1.4,
    "70+ hours": 1.5,
}

STUDY_FINE_TUNE_DAY0 = {
    "0 hours": -0.5,
    "1–2 hours": -0.2,
    "3–5 hours": 0.0,
    "6–10 hours": 0.35,
    "11–15 hours": 0.6,
    "16–20 hours": 0.85,
    "21–30 hours": 1.1,
    "31–40 hours": 1.35,
    "41–55 hours": 1.6,
    "56–70 hours": 1.8,
    "70+ hours": 2.0,
}

# default profile so we always send a complete row to the pipeline
# using pretty average values — not sure if this is the best approach but it works
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

# CSS — dark theme, tried to keep it clean
# the orange button was a last-minute design choice but it looks good tbh
CUSTOM_CSS = """
/* Center and widen main container */
.gradio-container {
    max-width: 1140px !important;
    margin: 0 auto !important;
    padding: 20px 24px !important;
}

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

/* cards for each form section */
.form-card {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    margin-bottom: 12px !important;
}

label span {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}

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

@media (max-width: 768px) {
    .gradio-container {
        padding: 12px !important;
    }
}
"""

def render_result_gauge(prob, pred_grade, status_text, status_color, risk_level, notes):
    # semi-circle gauge — radius 80, half circumference is 251.3
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
    sgpa,
    cgpa,
):
    study_tier = STUDY_HOURS_MAP.get(study_hours, 2)
    failures_num = int(failures)
    absences_num = int(absences)
    sgpa_num = float(sgpa)
    cgpa_num = float(cgpa)

    # GPA adjustment — both SGPA and CGPA shift the predicted grade
    # centered at 5.0 (midpoint), scaled so a 10 SGPA gives ~+0.6 and 0 gives ~-0.6
    gpa_boost = 0.40 * (sgpa_num - 5.0) / 5.0 + 0.25 * (cgpa_num - 5.0) / 5.0

    # start from the default profile and patch in what the user gave us
    data = DEFAULT_STUDENT_PROFILE.copy()
    data["studytime"] = study_tier
    data["failures"] = min(failures_num, 4)  # model was trained with max 4
    data["absences"] = absences_num
    data["paid"] = "yes" if paid_tutoring == "Yes" else "no"
    data["schoolsup"] = "yes" if academic_support == "Yes" else "no"

    notes = []

    if "Mid-Semester" in evaluation_stage:
        # scale scores up to 0-20 range that the model was trained on
        g1_scaled = (float(term1_score) / 10.0) * 20.0
        g2_scaled = (float(midterm_score) / 30.0) * 20.0
        data["G1"] = g1_scaled
        data["G2"] = g2_scaled

        df_input = pd.DataFrame([data])
        raw_prob = float(clf_midterm.predict_proba(df_input)[0][1]) * 100.0
        ml_grade = float(reg_midterm.predict(df_input)[0]) / 2.0  # bring 0-20 down to 0-10

        # weight the actual exam scores (term1=35%, midterm=50%)
        # this blending took some trial and error to get right
        exam_base = (0.35 * float(term1_score) + 0.50 * (float(midterm_score) / 3.0)) / 0.85
        combined_grade = 0.55 * exam_base + 0.45 * ml_grade

        study_bonus = STUDY_FINE_TUNE_MIDTERM.get(study_hours, 0.0)
        extra_fail = -0.25 * (failures_num - 4) if failures_num > 4 else 0.0
        pred_grade = np.clip(combined_grade + study_bonus + extra_fail + gpa_boost, 0.0, 10.0)

        # sigmoid centered at 5.0 (passing mark), then blend with raw model prob
        grade_prob = 100.0 / (1.0 + np.exp(-1.30 * (pred_grade - 5.0)))
        prob = 0.35 * raw_prob + 0.65 * grade_prob
        prob = float(np.clip(prob, 1.0, 99.5))

        notes.append(f"Evaluated with Term 1 ({term1_score}/10) and Mid-Term ({midterm_score}/30).")
        if pred_grade >= 7.0:
            notes.append("Internal exam scores demonstrate strong grasp of coursework.")
        elif pred_grade >= 5.0:
            notes.append("Internal scores are near passing cutoff (5.0/10); extra revision advised.")
        else:
            notes.append("Internal exam marks are currently below passing line (5.0/10); remedial action required.")

    else:
        # day-0 path — no exam scores available, going purely off background features
        df_input = pd.DataFrame([data])
        raw_prob = float(clf_day0.predict_proba(df_input)[0][1]) * 100.0
        ml_grade = float(reg_day0.predict(df_input)[0]) / 2.0  # scale 0-20 -> 0-10

        study_bonus = STUDY_FINE_TUNE_DAY0.get(study_hours, 0.0)
        extra_fail = -0.35 * (failures_num - 4) if failures_num > 4 else 0.0
        extra_abs = -0.02 * max(0, absences_num - 5)  # penalise absences above 5
        pred_grade = np.clip(ml_grade + study_bonus + extra_fail + extra_abs + gpa_boost, 0.0, 10.0)

        grade_prob = 100.0 / (1.0 + np.exp(-1.30 * (pred_grade - 5.0)))
        prob = 0.35 * raw_prob + 0.65 * grade_prob
        prob = float(np.clip(prob, 1.0, 99.5))

        notes.append("Day-1 enrollment evaluation based on academic background and study plan.")
        if pred_grade >= 7.0:
            notes.append("Baseline profile indicates high probability of semester success.")
        elif pred_grade >= 5.0:
            notes.append("Academic foundation is moderate; steady study habits will secure passing grade.")
        else:
            notes.append("Profile shows elevated risk; early tutoring and strict attendance advised.")

    # study habit notes
    if study_tier <= 1:
        notes.append(f"Weekly study time ({study_hours}) is low; increasing dedicated revision will improve results.")
    elif study_tier >= 3:
        notes.append(f"Consistent study schedule ({study_hours}) provides solid academic momentum.")

    if failures_num > 0:
        notes.append(f"Student has {failures_num} past backlog/failed subject(s) requiring remediation.")

    if absences_num > 8:
        notes.append(f"Recorded absences ({absences_num} days) elevate risk of missing foundational lecture concepts.")

    if paid_tutoring == "Yes" or academic_support == "Yes":
        notes.append("Academic tutoring / remedial support boosts student confidence and comprehension.")

    # GPA-based notes
    if sgpa_num >= 8.0:
        notes.append(f"Previous semester SGPA ({sgpa_num:.1f}/10) is strong — good academic momentum.")
    elif sgpa_num < 5.0:
        notes.append(f"Previous semester SGPA ({sgpa_num:.1f}/10) is below passing level — history of struggle.")

    if cgpa_num >= 7.5:
        notes.append(f"Overall CGPA ({cgpa_num:.1f}/10) indicates consistent academic performance.")
    elif cgpa_num < 5.0:
        notes.append(f"Overall CGPA ({cgpa_num:.1f}/10) suggests cumulative academic difficulties.")

    # figure out which status bucket the grade falls into
    if pred_grade >= 8.0:
        status_text = "HIGH PASS"
        status_color = "#10b981"  # green
        risk_level = "Very Low Risk"
    elif pred_grade >= 6.45:
        status_text = "PASS"
        status_color = "#10b981"
        risk_level = "Low Risk"
    elif pred_grade >= 5.0:
        status_text = "BORDERLINE PASS"
        status_color = "#f59e0b"  # amber
        risk_level = "Moderate Risk"
    else:
        status_text = "AT RISK OF FAILING"
        status_color = "#ef4444"  # red
        risk_level = "High Risk"

    return render_result_gauge(prob, pred_grade, status_text, status_color, risk_level, notes)

def toggle_exam_inputs(stage):
    is_midterm = "Mid-Semester" in stage
    return gr.update(visible=is_midterm)

# gradio layout
with gr.Blocks(css=CUSTOM_CSS, title="Student Academic Performance Prediction") as demo:
    with gr.Column(elem_classes="header-box"):
        gr.Markdown(
            """
            # Student Academic Performance Prediction
            Predict academic pass/fail risk and estimated final grade based on student habits and records.
            """
        )

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            with gr.Column(elem_classes="form-card"):
                evaluation_stage = gr.Radio(
                    [
                        "Semester Start (Day-1 Enrollment / Pre-Exam)",
                        "Mid-Semester (Includes Term 1 and Mid-Term Exam results)",
                    ],
                    value="Semester Start (Day-1 Enrollment / Pre-Exam)",
                    label="Evaluation Stage",
                )

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
                            "31–40 hours",
                            "41–55 hours",
                            "56–70 hours",
                            "70+ hours",
                        ],
                        value="6–10 hours",
                        label="Weekly Study Hours (out of 168h/week)",
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

                # only show exam inputs when mid-semester is selected
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

            with gr.Column(elem_classes="form-card"):
                gr.Markdown("**Academic Record**")
                with gr.Row():
                    sgpa = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=6.0,
                        step=0.1,
                        label="Previous Semester SGPA (0–10)",
                    )
                    cgpa = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=6.0,
                        step=0.1,
                        label="Overall CGPA (0–10)",
                    )

            submit_btn = gr.Button("Predict Performance", elem_id="predict-btn", size="lg")

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

    evaluation_stage.change(
        fn=toggle_exam_inputs,
        inputs=evaluation_stage,
        outputs=exam_scores_row,
    )

    input_components = [
        evaluation_stage,
        study_hours,
        failures,
        absences,
        term1_score,
        midterm_score,
        paid_tutoring,
        academic_support,
        sgpa,
        cgpa,
    ]

    submit_btn.click(
        fn=predict,
        inputs=input_components,
        outputs=result_output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
