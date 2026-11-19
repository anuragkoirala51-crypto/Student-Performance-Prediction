import os
import joblib
import pandas as pd
import gradio as gr

# Load pre-trained models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
clf_day0 = joblib.load(os.path.join(MODEL_DIR, "best_classifier_setup_d.joblib"))
clf_midterm = joblib.load(os.path.join(MODEL_DIR, "best_classifier_setup_a.joblib"))
reg_day0 = joblib.load(os.path.join(MODEL_DIR, "best_regressor_setup_d.joblib"))

# Mappings for user inputs
STUDY_HOURS = {
    "< 2 hours per week": 1,
    "2 to 5 hours per week": 2,
    "5 to 10 hours per week": 3,
    "> 10 hours per week": 4,
}

PARENT_EDU_MAP = {
    "Higher Education (Degree / College)": 4,
    "Secondary (High School / 12th)": 3,
    "Middle School (5th - 9th)": 2,
    "Primary (up to 4th)": 1,
    "None": 0,
}

def create_gauge_html(prob, pred_grade, status_text, status_color, eval_mode, advice_list):
    """Generate clean HTML for the circular score gauge and breakdown."""
    # Semi-circle gauge math: radius 80 -> half circumference is pi * 80 ~= 251.3
    arc_length = 251.3
    fill_offset = arc_length * (1.0 - (prob / 100.0))
    fill_offset = max(0, min(arc_length, fill_offset))

    advice_html = "".join([f"<li style='margin-bottom: 6px;'>{a}</li>" for a in advice_list])

    html = f"""
    <div style="background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 24px; color: #f9fafb; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
      
      <!-- Big Aesthetic Radial Gauge -->
      <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 12px;">
        <div style="position: relative; width: 240px; height: 140px;">
          <svg viewBox="0 0 200 120" style="width: 100%; height: 100%; overflow: visible;">
            <!-- Background track -->
            <path d="M 20 105 A 80 80 0 0 1 180 105" fill="none" stroke="#1f2937" stroke-width="16" stroke-linecap="round"/>
            <!-- Active colored arc -->
            <path d="M 20 105 A 80 80 0 0 1 180 105" fill="none" stroke="{status_color}" stroke-width="16" stroke-linecap="round"
                  stroke-dasharray="251.3" stroke-dashoffset="{fill_offset:.1f}" style="transition: stroke-dashoffset 0.8s ease;"/>
          </svg>
          <!-- Gauge Center Text -->
          <div style="position: absolute; top: 40px; left: 0; right: 0; text-align: center;">
            <div style="font-size: 42px; font-weight: 800; color: #ffffff; letter-spacing: -1px; line-height: 1;">{prob:.1f}%</div>
            <div style="font-size: 13px; font-weight: 700; letter-spacing: 1.5px; color: {status_color}; margin-top: 6px; text-transform: uppercase;">{status_text}</div>
          </div>
        </div>
      </div>

      <!-- Key Metrics Row -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px;">
        <div style="background: #1f2937; padding: 14px; border-radius: 12px; text-align: center; border: 1px solid #374151;">
          <div style="font-size: 12px; font-weight: 500; color: #9ca3af; text-transform: uppercase;">Predicted Final Grade</div>
          <div style="font-size: 24px; font-weight: 800; color: #38bdf8; margin-top: 4px;">{pred_grade:.1f} <span style="font-size: 14px; color: #9ca3af;">/ 20</span></div>
        </div>
        <div style="background: #1f2937; padding: 14px; border-radius: 12px; text-align: center; border: 1px solid #374151;">
          <div style="font-size: 12px; font-weight: 500; color: #9ca3af; text-transform: uppercase;">Evaluation Mode</div>
          <div style="font-size: 16px; font-weight: 700; color: #e5e7eb; margin-top: 8px;">{eval_mode}</div>
        </div>
      </div>

      <!-- Academic Insights -->
      <div style="margin-top: 16px; background: #1f2937; border-radius: 12px; padding: 16px; border: 1px solid #374151;">
        <div style="font-size: 13px; font-weight: 700; color: #e5e7eb; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Key Academic Insights</div>
        <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: #d1d5db; line-height: 1.5;">
          {advice_html}
        </ul>
      </div>

    </div>
    """
    return html

def predict(
    stage,
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
    # Standard healthy student baseline
    data = {
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

    advice = []

    # Choose model based on stage
    if "Mid-Semester" in stage:
        data["absences"] = int(absences)
        data["G1"] = int(g1_score)
        data["G2"] = int(g2_score)
        df_input = pd.DataFrame([data])
        prob = clf_midterm.predict_proba(df_input)[0][1] * 100
        pred_grade = 0.5 * (int(g1_score) + int(g2_score))
        mode_label = "Mid-Semester (With Internal Marks)"

        if int(g2_score) >= 12:
            advice.append("Consistent internal exam scores (G1 & G2) indicate high passing probability.")
        elif int(g2_score) < 10:
            advice.append("Mid-term exam scores are below pass threshold; remedial coaching recommended.")
    else:
        df_input = pd.DataFrame([data])
        prob = clf_day0.predict_proba(df_input)[0][1] * 100
        pred_grade = float(reg_day0.predict(df_input)[0])
        mode_label = "Day-1 Enrolment (Pre-Exam)"

        if STUDY_HOURS.get(study_time, 2) >= 3:
            advice.append(f"Strong self-study commitment ({study_time}) significantly boosts academic security.")
        else:
            advice.append("Increasing self-study time to 5+ hours weekly will reduce risk substantially.")

    if int(failures) > 0:
        advice.append(f"Student has {failures} past backlog(s) - monitoring and mentor check-ins advised.")
    else:
        advice.append("Clean academic record (0 past failures) strongly correlates with graduation.")

    if "Yes" in higher_ed:
        advice.append("Clear aspiration for higher education acts as a strong intrinsic motivator.")

    # Determine status and color
    if prob >= 70:
        status_text = "PASS (LOW RISK)"
        status_color = "#10b981"  # Emerald green
    elif prob >= 50:
        status_text = "PASS (MODERATE)"
        status_color = "#f59e0b"  # Amber / Yellow
    else:
        status_text = "AT RISK OF FAILING"
        status_color = "#ef4444"  # Red

    return create_gauge_html(prob, pred_grade, status_text, status_color, mode_label, advice)

# ----------------- Gradio Interface -----------------
with gr.Blocks(title="Student Academic Performance Prediction") as demo:
    gr.Markdown(
        """
        # Student Academic Performance Prediction
        ### Machine Learning Project — Early Warning System for Academic Outcome
        Configure student academic habits and background to evaluate pass probability and predicted final grade.
        """
    )

    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=3):
            stage = gr.Radio(
                [
                    "Semester Start (Day-1 Enrolment - Pre-Exam)",
                    "Mid-Semester (Includes Term 1 & 2 Internal Exams)",
                ],
                value="Semester Start (Day-1 Enrolment - Pre-Exam)",
                label="Evaluation Stage",
            )

            gr.Markdown("#### 📚 Academic Habits & Schooling")
            with gr.Row():
                study_time = gr.Dropdown(
                    list(STUDY_HOURS.keys()),
                    value="5 to 10 hours per week",
                    label="Weekly Study Time",
                )
                failures = gr.Slider(0, 4, value=0, step=1, label="Past Failed Classes / Backlogs")

            with gr.Row():
                school = gr.Radio(
                    ["GP - Gabriel Pereira", "MS - Mousinho da Silveira"],
                    value="GP - Gabriel Pereira",
                    label="School / Institution",
                )
                paid_classes = gr.Radio(["No", "Yes"], value="No", label="Extra Paid Tutoring")
                school_support = gr.Radio(["No", "Yes"], value="No", label="School Remedial Support")

            # Mid-term marks
            with gr.Row():
                absences = gr.Slider(0, 40, value=2, step=1, label="Semester Absences (Days)")
                g1_score = gr.Slider(0, 20, value=12, step=1, label="Term 1 Exam (G1: 0-20)")
                g2_score = gr.Slider(0, 20, value=13, step=1, label="Term 2 Exam (G2: 0-20)")

            gr.Markdown("#### 👤 Background & Lifestyle")
            with gr.Row():
                age = gr.Slider(15, 22, value=17, step=1, label="Student Age")
                gender = gr.Radio(["Female", "Male"], value="Female", label="Gender")
                area = gr.Radio(["Urban", "Rural"], value="Urban", label="Home Area")

            with gr.Row():
                parents_edu = gr.Dropdown(
                    list(PARENT_EDU_MAP.keys()),
                    value="Higher Education (Degree / College)",
                    label="Parents' Education Level",
                )
                higher_ed = gr.Radio(["Yes, planning for college", "No"], value="Yes, planning for college", label="Higher Education Goal")

            with gr.Row():
                internet = gr.Radio(["Yes", "No"], value="Yes", label="Internet Access at Home")
                goout = gr.Slider(1, 5, value=2, step=1, label="Social Outings (1: Low to 5: High)")
                alcohol = gr.Slider(1, 5, value=1, step=1, label="Weekend Alcohol / Partying (1-5)")
                fam_rel = gr.Slider(1, 5, value=4, step=1, label="Family Support & Relationship (1-5)")

            submit_btn = gr.Button("Evaluate Student Risk", variant="primary", size="lg")

        # Right Column: Big Aesthetic Result Gauge
        with gr.Column(scale=2):
            gr.Markdown("#### 📊 Evaluation Summary")
            result_output = gr.HTML(
                value="""
                <div style="background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 40px 20px; text-align: center; color: #9ca3af; font-family: sans-serif;">
                  <div style="font-size: 40px; margin-bottom: 10px;">📋</div>
                  <div style="font-size: 16px; font-weight: 600; color: #e5e7eb;">Ready to Evaluate</div>
                  <div style="font-size: 13px; margin-top: 6px;">Configure student details on the left and click <b>Evaluate Student Risk</b>.</div>
                </div>
                """
            )

    submit_btn.click(
        fn=predict,
        inputs=[
            stage,
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
