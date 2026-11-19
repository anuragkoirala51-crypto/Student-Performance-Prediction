# Simple Gradio app – beginner version (no AI)

import gradio as gr

# Helper map for study time
STUDY_MAP = {
    "<2 hrs/week": 1,
    "2‑5 hrs/week": 2,
    "5‑10 hrs/week": 3,
    ">10 hrs/week": 4,
}

def predict(study, failures, gender, age):
    """Very naive rule‑based prediction.

    The logic is deliberately simple so a beginner can see how the
    calculation works.
    """
    # Start with a base score
    score = 0
    # Study time contributes positively
    score += STUDY_MAP.get(study, 0) * 10
    # More past failures hurt the score
    score -= int(failures) * 15
    # Small gender bonus (just for demo purposes)
    if gender == "Female":
        score += 5
    # Younger students get a small bump
    score += max(0, 25 - int(age))
    # Clamp to 0‑100
    prob = max(0, min(100, score))
    status = "PASS" if prob >= 50 else "FAIL / AT RISK"
    return f"Status: {status}\nPass probability: {prob}%"

# Simple UI – four inputs only
with gr.Blocks() as demo:
    gr.Markdown("# Student Performance Predictor (Beginner, No AI)")
    study = gr.Dropdown(list(STUDY_MAP.keys()), label="Study time per week")
    failures = gr.Slider(0, 3, step=1, label="Number of past failures")
    gender = gr.Radio(["Male", "Female"], label="Gender")
    age = gr.Slider(15, 25, step=1, label="Age")
    btn = gr.Button("Predict")
    out = gr.Textbox(label="Result")
    btn.click(predict, inputs=[study, failures, gender, age], outputs=out)

if __name__ == "__main__":
    # Local launch; set share=True for a public link
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
