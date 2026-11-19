# Simple Gradio app – beginner version

import os
import joblib
import pandas as pd
import gradio as gr

# Load pre‑trained classifier (kept the same path as before)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
classifier = joblib.load(os.path.join(MODEL_DIR, "best_classifier_setup_d.joblib"))

# Very small helper maps – students can understand them easily
STUDY_MAP = {
    "<2 hrs/week": 1,
    "2‑5 hrs/week": 2,
    "5‑10 hrs/week": 3,
    ">10 hrs/week": 4,
}


def predict(study, failures, gender, age):
    """Create a tiny feature set and return a pass / fail guess.

    Args:
        study (str): chosen study‑time bucket.
        failures (int): number of past failed subjects.
        gender (str): "Male" or "Female".
        age (int): student age.
    """
    # Build a minimal dataframe – most columns are fixed dummy values
    data = {
        "school": "GP",
        "sex": "M" if gender == "Male" else "F",
        "age": int(age),
        "studytime": STUDY_MAP[study],
        "failures": int(failures),
        # Fixed values just to satisfy the model’s expected columns
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 2,
        "Fedu": 2,
        "Mjob": "services",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "schoolsup": "no",
        "famsup": "no",
        "paid": "no",
        "activities": "no",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 2,
        "Dalc": 1,
        "Walc": 2,
        "health": 4,
    }
    df = pd.DataFrame([data])
    prob = classifier.predict_proba(df)[0][1]
    status = "PASS" if prob >= 0.5 else "FAIL / AT RISK"
    return f"Status: {status}\nPass probability: {prob*100:.1f}%"

# Very small UI – only four inputs
with gr.Blocks() as demo:
    gr.Markdown("# Student Performance Predictor (Beginner Version)")
    study = gr.Dropdown(list(STUDY_MAP.keys()), label="Study time per week")
    failures = gr.Slider(0, 3, step=1, label="Number of past failures")
    gender = gr.Radio(["Male", "Female"], label="Gender")
    age = gr.Slider(15, 25, step=1, label="Age")
    btn = gr.Button("Predict")
    out = gr.Textbox(label="Result")
    btn.click(predict, inputs=[study, failures, gender, age], outputs=out)

if __name__ == "__main__":
    # keep it local; change share=True if you need a public link
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
