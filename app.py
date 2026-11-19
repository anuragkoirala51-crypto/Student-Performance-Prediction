import os
import joblib
import pandas as pd
import gradio as gr

# Load models saved by the training pipeline
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
clf_day0 = joblib.load(os.path.join(MODEL_DIR, "best_classifier_setup_d.joblib"))
clf_midterm = joblib.load(os.path.join(MODEL_DIR, "best_classifier_setup_a.joblib"))
reg_day0 = joblib.load(os.path.join(MODEL_DIR, "best_regressor_setup_d.joblib"))

# Input mapping dictionaries
STUDY_HOURS = {
    "< 2 hours per week": 1,
    "2 to 5 hours per week": 2,
    "5 to 10 hours per week": 3,
    "> 10 hours per week": 4,
}

PARENT_EDU_MAP = {
    "Higher Education (College / Degree)": 4,
    "Secondary (10th / 12th)": 3,
    "Middle School (5th to 8th)": 2,
    "Primary (up to 4th)": 1,
    "None": 0,
}

def render_result_gauge(prob, pred_grade, status_text, status_color, notes):
    """Draw circular score meter and summary card."""
    # Semi-circle gauge (radius 80, half circumference ~ 251.3)
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
            <!-- Background Arc -->
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

def predict_student(
    evaluation_stage,
    study_time,
    failures,
    school,
    paid_classes,
    school_support,
    absences,
    g1_score,
    g2_score,
    age,
    gender,
    area,
    parents_edu,
    higher_ed,
    internet,
    goout,
    alcohol,
    fam_rel,
):
    # Construct feature dictionary for model pipeline
    student_data = {
        "school": "GP" if "GP" in school else "MS",
        "sex": "F" if gender == "Female" else "M",
        "age": int(age),
        "address": "U" if area == "Urban" else "R",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": PARENT_EDU_MAP.get(parents_edu, 3),
        "Fedu": PARENT_EDU_MAP.get(parents_edu, 3),
        "Mjob": "services",
        "Fjob": "services",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": STUDY_HOURS.get(study_time, 2),
        "failures": int(failures),
        "schoolsup": "yes" if school_support == "Yes" else "no",
        "famsup": "yes",
        "paid": "yes" if paid_classes == "Yes" else "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes" if "Yes" in higher_ed else "no",
        "internet": "yes" if internet == "Yes" else "no",
        "romantic": "no",
        "famrel": int(fam_rel),
        "freetime": 3,
        "goout": int(goout),
        "Dalc": 1,
        "Walc": int(alcohol),
        "health": 4,
    }

    notes = []

    # Run appropriate model based on stage
    if "Mid-Semester" in evaluation_stage:
        student_data["absences"] = int(absences)
        student_data["G1"] = int(g1_score)
        student_data["G2"] = int(g2_score)
        df_row = pd.DataFrame([student_data])
        prob = clf_midterm.predict_proba(df_row)[0][1] * 100
        pred_grade = 0.5 * (int(g1_score) + int(g2_score))

        if int(g2_score) >= 12:
            notes.append("Good scores in internal exams (G1 and G2).")
        elif int(g2_score) < 10:
            notes.append("Internal exam marks are currently below passing threshold (10/20).")
    else:
        df_row = pd.DataFrame([student_data])
        prob = clf_day0.predict_proba(df_row)[0][1] * 100
        pred_grade = float(reg_day0.predict(df_row)[0])

        if STUDY_HOURS.get(study_time, 2) >= 3:
            notes.append(f"Regular weekly study hours ({study_time}).")
        else:
            notes.append("Study time is relatively low; increasing study hours will improve chances.")

    if int(failures) == 0:
        notes.append("No past failed subjects or backlogs.")
    else:
        notes.append(f"Student has {failures} past backlog(s).")

    if int(absences) > 10:
        notes.append(f"High number of absences ({absences} days).")

    # Set status badge and color
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

# ----------------- Gradio UI -----------------
with gr.Blocks(title="Student Performance Prediction") as demo:
    gr.Markdown(
        """
        # Student Academic Performance Prediction
        Enter student details and study habits to predict pass/fail result and expected final grade.
        """
    )

    with gr.Row():
        # Input Form
        with gr.Column(scale=3):
            evaluation_stage = gr.Radio(
                [
                    "Semester Start (Before internal exams)",
                    "Mid-Semester (With Term 1 & 2 exam marks)",
                ],
                value="Semester Start (Before internal exams)",
                label="Evaluation Timing",
            )

            gr.Markdown("### Academic Information")
            with gr.Row():
                study_time = gr.Dropdown(
                    list(STUDY_HOURS.keys()),
                    value="5 to 10 hours per week",
                    label="Weekly Study Time",
                )
                failures = gr.Slider(0, 4, value=0, step=1, label="Past Failed Subjects")

            with gr.Row():
                school = gr.Radio(
                    ["GP - Gabriel Pereira", "MS - Mousinho da Silveira"],
                    value="GP - Gabriel Pereira",
                    label="School Name",
                )
                paid_classes = gr.Radio(["No", "Yes"], value="No", label="Extra Tuition / Coaching")
                school_support = gr.Radio(["No", "Yes"], value="No", label="Remedial Classes")

            with gr.Row():
                absences = gr.Slider(0, 40, value=2, step=1, label="Absences (Days Missed)")
                g1_score = gr.Slider(0, 20, value=12, step=1, label="Internal Exam 1 Marks (0-20)")
                g2_score = gr.Slider(0, 20, value=13, step=1, label="Internal Exam 2 Marks (0-20)")

            gr.Markdown("### Personal Background & Habits")
            with gr.Row():
                age = gr.Slider(15, 22, value=17, step=1, label="Age")
                gender = gr.Radio(["Female", "Male"], value="Female", label="Gender")
                area = gr.Radio(["Urban", "Rural"], value="Urban", label="Area (Urban / Rural)")

            with gr.Row():
                parents_edu = gr.Dropdown(
                    list(PARENT_EDU_MAP.keys()),
                    value="Higher Education (College / Degree)",
                    label="Parents' Education",
                )
                higher_ed = gr.Radio(["Yes", "No"], value="Yes", label="Plans for Higher Studies")

            with gr.Row():
                internet = gr.Radio(["Yes", "No"], value="Yes", label="Internet Access at Home")
                goout = gr.Slider(1, 5, value=2, step=1, label="Going Out with Friends (1 to 5)")
                alcohol = gr.Slider(1, 5, value=1, step=1, label="Weekend Alcohol / Partying (1 to 5)")
                fam_rel = gr.Slider(1, 5, value=4, step=1, label="Family Support (1 to 5)")

            submit_btn = gr.Button("Predict Performance", variant="primary", size="lg")

        # Result Display (Gauge & Cards)
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

    submit_btn.click(
        fn=predict_student,
        inputs=[
            evaluation_stage,
            study_time,
            failures,
            school,
            paid_classes,
            school_support,
            absences,
            g1_score,
            g2_score,
            age,
            gender,
            area,
            parents_edu,
            higher_ed,
            internet,
            goout,
            alcohol,
            fam_rel,
        ],
        outputs=result_output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
