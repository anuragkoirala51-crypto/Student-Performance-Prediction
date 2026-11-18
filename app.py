"""
Student Performance Prediction System
-------------------------------------
Machine Learning Coursework Project
Author: Anurag Koirala (Roll No: ECB24030)
Dataset: UCI Student Performance (Cortez, 2008)
"""

import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Load models
MODEL_DIR = "models"
pipe_d = joblib.load(os.path.join(MODEL_DIR, "best_classifier_setup_d.joblib"))
reg_d = joblib.load(os.path.join(MODEL_DIR, "best_regressor_setup_d.joblib"))
pipe_a = joblib.load(os.path.join(MODEL_DIR, "best_classifier_setup_a.joblib"))

# Mappings
STUDY_MAP = {
    "Less than 2 hours per week": 1,
    "2 to 5 hours per week": 2,
    "5 to 10 hours per week": 3,
    "More than 10 hours per week": 4
}

GOOUT_MAP = {
    "Rarely (Level 1/5)": 1,
    "Occasionally (Level 2/5)": 2,
    "Moderate (Level 3/5)": 3,
    "Frequently (Level 4/5)": 4,
    "Very Frequently (Level 5/5)": 5
}

def predict_performance(
    timeline_mode, study_text, failures_count, goout_text, paid_help, wants_college,
    midterm_g1, midterm_g2, absences_days,
    student_age, student_gender, student_area, school_support, family_support,
    extra_activities, internet_home, family_relation
):
    studytime = STUDY_MAP.get(study_text, 2)
    goout = GOOUT_MAP.get(goout_text, 3)
    paid = "yes" if "Yes" in paid_help else "no"
    higher = "yes" if "Yes" in wants_college else "no"
    sex = "F" if "Female" in student_gender else "M"
    address = "U" if "Urban" in student_area else "R"
    schoolsup = "yes" if "Yes" in school_support else "no"
    famsup = "yes" if "Yes" in family_support else "no"
    activities = "yes" if "Yes" in extra_activities else "no"
    internet = "yes" if "Yes" in internet_home else "no"

    data = {
        "school": "GP",
        "sex": sex,
        "age": int(student_age),
        "address": address,
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 2,
        "Fedu": 2,
        "Mjob": "services",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": studytime,
        "failures": int(failures_count),
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": "yes",
        "higher": higher,
        "internet": internet,
        "romantic": "no",
        "famrel": int(family_relation),
        "freetime": 3,
        "goout": goout,
        "Dalc": 1,
        "Walc": 2,
        "health": 4
    }

    is_day0 = ("Semester Start" in timeline_mode or "Enrolment" in timeline_mode or "Day-1" in timeline_mode or "Setup D" in timeline_mode)

    if not is_day0:
        data["absences"] = int(absences_days)
        data["G1"] = int(midterm_g1)
        data["G2"] = int(midterm_g2)
        model = pipe_a
    else:
        model = pipe_d

    df_student = pd.DataFrame([data])

    probs = model.predict_proba(df_student)[0]
    p_pass = float(probs[1])
    p_fail = float(probs[0])

    if is_day0:
        predicted_grade = float(reg_d.predict(df_student)[0])
    else:
        predicted_grade = 0.5 * int(midterm_g1) + 0.5 * int(midterm_g2)
    predicted_grade = max(0.0, min(20.0, predicted_grade))

    pass_pct = p_pass * 100
    fail_pct = p_fail * 100

    if p_pass >= 0.50:
        status_label = "PASS"
        text_color = "#34d399"
        border_color = "#10b981"
        badge_bg = "rgba(16, 185, 129, 0.15)"
        desc = "The model predicts the student is on track to pass the course (Final Grade >= 10)."
    else:
        status_label = "FAIL / AT RISK"
        text_color = "#f87171"
        border_color = "#ef4444"
        badge_bg = "rgba(239, 68, 68, 0.15)"
        desc = "The model predicts the student is at risk of failing the course (Final Grade < 10) and needs academic support."

    summary_card_html = f"""
    <div style='background: #111827; border: 1.5px solid {border_color}; border-radius: 12px; padding: 22px; box-shadow: 0 4px 16px rgba(0,0,0,0.3); color: #f8fafc;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;'>
            <div style='font-size: 13px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;'>
                Prediction Result
            </div>
            <div style='background: {badge_bg}; color: {text_color}; border: 1px solid {border_color}; padding: 4px 14px; border-radius: 20px; font-weight: 800; font-size: 13px;'>
                {status_label}
            </div>
        </div>

        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px;'>
            <div style='background: #0f172a; border: 1px solid #374151; padding: 16px; border-radius: 10px; text-align: center;'>
                <div style='font-size: 12px; color: #94a3b8; margin-bottom: 4px;'>Pass Probability</div>
                <div style='font-size: 32px; font-weight: 800; color: {text_color};'>{pass_pct:.0f}%</div>
                <div style='font-size: 11px; color: #64748b; margin-top: 2px;'>Fail Probability: {fail_pct:.0f}%</div>
            </div>
            <div style='background: #0f172a; border: 1px solid #374151; padding: 16px; border-radius: 10px; text-align: center;'>
                <div style='font-size: 12px; color: #94a3b8; margin-bottom: 4px;'>Expected Final Score (G3)</div>
                <div style='font-size: 32px; font-weight: 800; color: #38bdf8;'>{predicted_grade:.1f} <span style='font-size: 16px; font-weight: 500; color: #64748b;'>/ 20</span></div>
                <div style='font-size: 11px; color: #64748b; margin-top: 2px;'>Pass Mark: 10.0 / 20</div>
            </div>
        </div>

        <div style='background: #0f172a; border-left: 4px solid {border_color}; padding: 12px 16px; border-radius: 8px;'>
            <div style='font-size: 12px; font-weight: 700; color: #e2e8f0; margin-bottom: 2px;'>Evaluation Note</div>
            <div style='font-size: 13px; color: #94a3b8; line-height: 1.5;'>{desc}</div>
        </div>
    </div>
    """

    study_scenarios = [
        ("Less than 2 hours per week", 1),
        ("2 to 5 hours per week", 2),
        ("5 to 10 hours per week", 3),
        ("More than 10 hours per week", 4)
    ]
    comparison_cards_html = "<div style='display: flex; flex-direction: column; gap: 8px; margin-top: 8px;'>"
    for label, val in study_scenarios:
        temp_data = df_student.copy()
        temp_data["studytime"] = val
        sim_p_pass = float(model.predict_proba(temp_data)[0][1])
        is_current = (val == studytime)

        if is_current:
            border = "border: 2px solid #38bdf8; background: #0c4a6e;"
            badge = "<span style='background: #0284c7; color: white; padding: 2px 8px; border-radius: 6px; font-size: 11px; margin-left: 8px; font-weight: 600;'>Selected</span>"
        else:
            border = "border: 1px solid #374151; background: #111827;"
            badge = ""

        comparison_cards_html += f"""
        <div style='{border} padding: 12px 16px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center; color: #f8fafc;'>
            <div style='font-size: 13.5px;'>
                <strong>{label}</strong> {badge}
            </div>
            <div>
                <span style='font-size: 14px; font-weight: 700; color: #38bdf8;'>{sim_p_pass * 100:.0f}% Pass Chance</span>
            </div>
        </div>
        """
    comparison_cards_html += "</div>"

    return summary_card_html, comparison_cards_html

# Default precomputed values for instant rendering
INITIAL_SUMMARY, INITIAL_COMPARISON = predict_performance(
    "Semester Start (Predict before any exams have taken place)",
    "2 to 5 hours per week",
    0,
    "Moderate (Level 3/5)",
    "No",
    "Yes, planning for higher studies",
    11, 11, 4,
    18, "Female", "Urban", "No", "Yes", "No", "Yes", 4
)

CUSTOM_CSS = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #f8fafc !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

.gradio-container {
    max-width: 1140px !important;
    margin: 0 auto !important;
    padding: 24px 16px !important;
}

.custom-header {
    background: #111827;
    border: 1px solid #1f2937;
    padding: 20px 24px;
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}

.custom-footer {
    margin-top: 24px;
    padding: 16px 20px;
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.5;
    text-align: center;
}
"""

theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="blue",
    neutral_hue="slate"
)

with gr.Blocks(title="Student Performance Prediction System", theme=theme, css=CUSTOM_CSS) as demo:
    # ----------------- HEADER & STUDENT CREDENTIALS -----------------
    with gr.Row(elem_classes="custom-header"):
        gr.HTML("""
        <div style='width: 100%; display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;'>
            <div>
                <div style='font-size: 12px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px;'>
                    College Coursework Project &bull; Machine Learning
                </div>
                <h1 style='font-size: 1.6rem; font-weight: 700; color: #f8fafc; margin: 0 0 6px 0; letter-spacing: -0.2px;'>
                    Student Academic Performance Prediction System
                </h1>
                <div style='font-size: 13.5px; color: #94a3b8;'>
                    Predicting pass/fail risk and expected marks using student study habits and academic history
                </div>
            </div>
            <div style='background: #0f172a; border: 1px solid #1e293b; padding: 10px 16px; border-radius: 8px; text-align: right;'>
                <div style='font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;'>Submitted by</div>
                <div style='font-size: 14px; font-weight: 700; color: #f8fafc;'>Anurag Koirala</div>
                <div style='font-size: 12px; color: #38bdf8; font-weight: 600;'>Roll No: ECB24030</div>
            </div>
        </div>
        """)

    # ----------------- MAIN LAYOUT -----------------
    with gr.Row(equal_height=False):
        # LEFT COLUMN: INPUTS
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 1. Select Prediction Stage")
                timeline_mode = gr.Radio(
                    choices=[
                        "Semester Start (Predict before any exams have taken place)",
                        "Mid-Semester (Include Term 1 & 2 internal exam marks)"
                    ],
                    value="Semester Start (Predict before any exams have taken place)",
                    label="When is this evaluation taking place?"
                )

                gr.Markdown("<div style='height: 8px;'></div>")
                gr.Markdown("### 2. Student Study & Academic Habits")
                study_text = gr.Dropdown(
                    choices=list(STUDY_MAP.keys()),
                    value="2 to 5 hours per week",
                    label="Weekly Self-Study Time"
                )
                failures_count = gr.Slider(
                    minimum=0, maximum=3, step=1, value=0,
                    label="Past Failed Subjects / Backlogs (0, 1, 2, or 3+)"
                )
                goout_text = gr.Dropdown(
                    choices=list(GOOUT_MAP.keys()),
                    value="Moderate (Level 3/5)",
                    label="Socializing & Outings with Friends"
                )
                with gr.Row():
                    paid_help = gr.Radio(
                        ["No", "Yes, extra tutoring"],
                        value="No",
                        label="Extra Coaching / Tuition Classes"
                    )
                    wants_college = gr.Radio(
                        ["Yes, planning for higher studies", "No"],
                        value="Yes, planning for higher studies",
                        label="Plans for Higher Studies (Master's / Postgrad)"
                    )

                with gr.Accordion("Internal Exam Marks & Attendance (Only used in Mid-Semester mode)", open=False):
                    gr.Markdown("<p style='font-size: 12px; color: #94a3b8;'>Ignored when Semester Start is selected.</p>")
                    with gr.Row():
                        midterm_g1 = gr.Slider(0, 20, value=11, step=1, label="Internal Exam 1 (G1: 0-20)")
                        midterm_g2 = gr.Slider(0, 20, value=11, step=1, label="Internal Exam 2 (G2: 0-20)")
                    absences_days = gr.Slider(0, 50, value=4, step=1, label="Semester Absences (Class Days Missed: 0-50)")

                with gr.Accordion("Student Background & College Life (Optional)", open=False):
                    with gr.Row():
                        student_age = gr.Slider(15, 25, value=18, step=1, label="Student Age (Years)")
                        student_gender = gr.Radio(["Female", "Male"], value="Female", label="Gender")
                    with gr.Row():
                        student_area = gr.Radio(["Urban", "Rural"], value="Urban", label="Home Area")
                        internet_home = gr.Radio(["Yes", "No"], value="Yes", label="Internet at Home / Hostel")
                    with gr.Row():
                        school_support = gr.Radio(["No", "Yes"], value="No", label="College Remedial Academic Support")
                        family_support = gr.Radio(["Yes", "No"], value="Yes", label="Family Support")
                    with gr.Row():
                        extra_activities = gr.Radio(["No", "Yes"], value="No", label="College Clubs & Extracurricular Activities")
                        family_relation = gr.Slider(1, 5, value=4, step=1, label="Family Support & Relationship (1-5)")

                predict_btn = gr.Button("Predict Student Result", variant="primary")

        # RIGHT COLUMN: RESULTS
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Prediction Result")
                summary_output = gr.HTML(value=INITIAL_SUMMARY)

                gr.Markdown("<div style='height: 14px;'></div>")
                gr.Markdown("### Study Hours vs. Predicted Pass Chance")
                gr.Markdown("<p style='font-size: 12.5px; color: #94a3b8; margin-top: -6px;'>See how increasing weekly self-study hours affects the pass probability:</p>")
                comparison_output = gr.HTML(value=INITIAL_COMPARISON)

    gr.HTML("""
    <div class='custom-footer'>
        <strong>College Coursework Project:</strong> Submitted by <strong>Anurag Koirala (Roll No: ECB24030)</strong>. Built with Scikit-Learn (Random Forest) and Gradio using the UCI Student Performance Dataset.
    </div>
    """)

    inputs_list = [
        timeline_mode, study_text, failures_count, goout_text, paid_help, wants_college,
        midterm_g1, midterm_g2, absences_days,
        student_age, student_gender, student_area, school_support, family_support,
        extra_activities, internet_home, family_relation
    ]
    predict_btn.click(
        fn=predict_performance,
        inputs=inputs_list,
        outputs=[summary_output, comparison_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
