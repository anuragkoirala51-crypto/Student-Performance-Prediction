import os
import joblib
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

# Neutral default background features to ensure the ML pipeline receives valid inputs
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

def render_result_gauge(prob, pred_grade, status_text, status_color, notes):
    """Render circular score meter and summary card in dark theme."""
    # Semi-circle gauge math (radius 80, half circumference = 251.3)
    arc_len = 251.3
    offset = arc_len * (1.0 - (prob / 100.0))
    offset = max(0, min(arc_len, offset))

    note_items = "".join([f"<li style='margin-bottom: 5px;'>{n}</li>" for n in notes])

    html = f"""
    <div style="background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 22px; color: #f3f4f6; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
      
      <!-- Semi-Circular Progress Meter -->
      <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <div style="position: relative; width: 230px; height: 135px;">
          <svg viewBox="0 0 200 120" style="width: 100%; height: 100%; overflow: visible;">
            <!-- Background Track -->
            <path d="M 20 105 A 80 80 0 0 1 180 105" fill="none" stroke="#1f2937" stroke-width="16" stroke-linecap="round"/>
            <!-- Value Arc -->
            <path d="M 20 105 A 80 80 0 0 1 180 105" fill="none" stroke="{status_color}" stroke-width="16" stroke-linecap="round"
                  stroke-dasharray="251.3" stroke-dashoffset="{offset:.1f}"/>
          </svg>
          <div style="position: absolute; top: 40px; left: 0; right: 0; text-align: center;">
            <div style="font-size: 40px; font-weight: 800; color: #ffffff; line-height: 1;">{prob:.1f}%</div>
            <div style="font-size: 13px; font-weight: 700; letter-spacing: 1.2px; color: {status_color}; margin-top: 5px;">{status_text}</div>
          </div>
        </div>
      </div>

      <!-- Quick Metrics -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px;">
        <div style="background: #1f2937; padding: 12px; border-radius: 10px; text-align: center; border: 1px solid #374151;">
          <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase;">Expected Final Score</div>
          <div style="font-size: 22px; font-weight: 700; color: #38bdf8; margin-top: 3px;">{pred_grade:.1f} <span style="font-size: 13px; color: #9ca3af;">/ 20</span></div>
        </div>
        <div style="background: #1f2937; padding: 12px; border-radius: 10px; text-align: center; border: 1px solid #374151;">
          <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase;">Prediction Status</div>
          <div style="font-size: 18px; font-weight: 700; color: {status_color}; margin-top: 6px;">{status_text}</div>
        </div>
      </div>

      <!-- Summary Observations -->
      <div style="margin-top: 14px; background: #1f2937; border-radius: 10px; padding: 14px; border: 1px solid #374151;">
        <div style="font-size: 12px; font-weight: 700; color: #e5e7eb; margin-bottom: 6px; text-transform: uppercase;">Summary & Observations</div>
        <ul style="margin: 0; padding-left: 18px; font-size: 13px; color: #d1d5db; line-height: 1.45;">
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
    # Initialize features from baseline profile
    data = DEFAULT_STUDENT_PROFILE.copy()
    data["studytime"] = STUDY_HOURS_MAP.get(study_hours, 2)
    data["failures"] = min(int(failures), 4)
    data["paid"] = "yes" if paid_tutoring == "Yes" else "no"
    data["schoolsup"] = "yes" if academic_support == "Yes" else "no"

    notes = []

    # Dynamic behavior based on Evaluation Stage
    if "Mid-Semester" in evaluation_stage:
        # Scale Term 1 (out of 10) and Mid-Term (out of 30) to the model's 0-20 scale
        g1_scaled = (float(term1_score) / 10.0) * 20.0
        g2_scaled = (float(midterm_score) / 30.0) * 20.0

        data["absences"] = int(absences)
        data["G1"] = g1_scaled
        data["G2"] = g2_scaled

        df_input = pd.DataFrame([data])
        prob = clf_midterm.predict_proba(df_input)[0][1] * 100
        pred_grade = 0.5 * (g1_scaled + g2_scaled)

        notes.append(f"Term 1: {term1_score}/10 | Mid-Term: {midterm_score}/30 included in evaluation.")
        if g2_scaled >= 12:
            notes.append("Strong internal exam scores indicate solid passing trajectory.")
        elif g2_scaled < 10:
            notes.append("Internal exam scores are below passing grade (10/20); extra support advised.")
    else:
        # Semester Start: Use Day-0 models which do not require or assume G1/G2
        df_input = pd.DataFrame([data])
        prob = clf_day0.predict_proba(df_input)[0][1] * 100
        pred_grade = float(reg_day0.predict(df_input)[0])

        notes.append("Day-1 enrollment evaluation based on academic habits and background.")
        if STUDY_HOURS_MAP.get(study_hours, 2) >= 3:
            notes.append(f"Committed study schedule ({study_hours}) provides strong academic stability.")
        else:
            notes.append("Study hours are relatively low; committing 6+ hours weekly will improve performance.")

    if int(failures) == 0:
        notes.append("Clean academic record with no previous failed classes.")
    else:
        notes.append(f"Student has {failures} previous backlog/failed subject(s).")

    if int(absences) > 8:
        notes.append(f"Recorded absences ({absences} days) could impact end-of-term results.")

    # Status classification
    if prob >= 70:
        status_text = "PASS"
        status_color = "#10b981"  # Green
    elif prob >= 50:
        status_text = "BORDERLINE PASS"
        status_color = "#f59e0b"  # Amber
    else:
        status_text = "AT RISK OF FAILING"
        status_color = "#ef4444"  # Red

    return render_result_gauge(prob, pred_grade, status_text, status_color, notes)

def toggle_exam_inputs(stage):
    """Show or hide exam score inputs depending on the selected stage."""
    is_midterm = "Mid-Semester" in stage
    return gr.update(visible=is_midterm)

# ----------------- Gradio UI -----------------
with gr.Blocks(title="Student Academic Performance Prediction") as demo:
    gr.Markdown(
        """
        # Student Academic Performance Prediction
        Enter student details to evaluate academic risk and predict final performance.
        """
    )

    with gr.Row():
        # Input Form Column
        with gr.Column(scale=3):
            evaluation_stage = gr.Radio(
                [
                    "Semester Start (Day-1 Enrollment / Pre-Exam)",
                    "Mid-Semester (Includes Term 1 and Mid-Term Exam results)",
                ],
                value="Semester Start (Day-1 Enrollment / Pre-Exam)",
                label="Evaluation Stage",
            )

            gr.Markdown("### Student Academic Information")
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

            with gr.Row():
                absences = gr.Slider(
                    minimum=0,
                    maximum=30,
                    value=0,
                    step=1,
                    label="Semester Absences (Days)",
                )

            # Exam Scores Row (dynamically visible only in Mid-Semester mode)
            with gr.Row(visible=False) as exam_scores_row:
                term1_score = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Term 1 Exam Score (out of 10 marks)",
                )
                midterm_score = gr.Slider(
                    minimum=0,
                    maximum=30,
                    value=15,
                    step=1,
                    label="Mid-Term Exam Score (out of 30 marks)",
                )

            with gr.Row():
                paid_tutoring = gr.Radio(
                    ["No", "Yes"],
                    value="No",
                    label="Extra Paid Tutoring",
                )
                academic_support = gr.Radio(
                    ["No", "Yes"],
                    value="No",
                    label="Academic Support / Remedial Classes",
                )

            submit_btn = gr.Button("Predict Performance", variant="primary", size="lg")

        # Result Display Column (Aesthetic Gauge Card)
        with gr.Column(scale=2):
            gr.Markdown("### Prediction Result")
            result_output = gr.HTML(
                value="""
                <div style="background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 35px 20px; text-align: center; color: #9ca3af; font-family: sans-serif;">
                  <div style="font-size: 16px; font-weight: 600; color: #e5e7eb;">Ready to Predict</div>
                  <div style="font-size: 13px; margin-top: 6px;">Fill in the student details on the left and click <b>Predict Performance</b>.</div>
                </div>
                """
            )

    # Dynamic visibility toggle when evaluation stage changes
    evaluation_stage.change(
        fn=toggle_exam_inputs,
        inputs=evaluation_stage,
        outputs=exam_scores_row,
    )

    # Prediction triggers
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
