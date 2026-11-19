import joblib
import pandas as pd
import gradio as gr

# Load trained machine learning model
model = joblib.load("models/best_classifier_setup_d.joblib")
template = pd.read_csv("data/sample_student.csv").iloc[0].to_dict()

def predict_student(gender, age, studytime, failures, goout, internet, higher):
    # Prepare feature row
    data = template.copy()
    data["sex"] = "M" if gender == "Male" else "F"
    data["age"] = int(age)
    data["studytime"] = int(studytime)
    data["failures"] = int(failures)
    data["goout"] = int(goout)
    data["internet"] = "yes" if internet == "Yes" else "no"
    data["higher"] = "yes" if higher == "Yes" else "no"

    df = pd.DataFrame([data])
    prob_pass = model.predict_proba(df)[0][1]

    if prob_pass >= 0.5:
        return f"Result: PASS\nPass Probability: {prob_pass * 100:.1f}%\nStatus: On Track"
    else:
        return f"Result: AT RISK OF FAILING\nPass Probability: {prob_pass * 100:.1f}%\nStatus: Needs Academic Support"

# Inputs for student evaluation
inputs = [
    gr.Radio(["Female", "Male"], label="Gender", value="Female"),
    gr.Slider(15, 22, value=17, step=1, label="Age"),
    gr.Slider(1, 4, value=2, step=1, label="Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)"),
    gr.Slider(0, 4, value=0, step=1, label="Past Failures (0 to 4)"),
    gr.Slider(1, 5, value=3, step=1, label="Going Out with Friends (1: Low to 5: High)"),
    gr.Radio(["Yes", "No"], label="Internet Access", value="Yes"),
    gr.Radio(["Yes", "No"], label="Plans for Higher Education", value="Yes"),
]

demo = gr.Interface(
    fn=predict_student,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction Output", lines=4),
    title="Student Academic Performance Prediction",
    description="Machine Learning assignment project to predict student pass/fail risk based on habits and background."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
