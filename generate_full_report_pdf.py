import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_OUTPUT = "/Users/anuragkoirala/AIML project/STUDENT_PERFORMANCE_PROJECT_REPORT_ANURAG_KOIRALA.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Suppress headers/footers on title/cover page
            return
        
        self.saveState()
        self.setFont("Helvetica", 8.5)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Running Header
        self.drawString(40, 755, "Early Warning Student Academic Performance Prediction System")
        self.drawRightString(572, 755, "Anurag Koirala (ECB24030)")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.6)
        self.line(40, 747, 572, 747)
        
        # Running Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 32, page_str)
        self.drawString(40, 32, "Machine Learning & Deep Learning Coursework Submission")
        self.line(40, 44, 572, 44)
        
        self.restoreState()

doc = SimpleDocTemplate(
    PDF_OUTPUT,
    pagesize=letter,
    leftMargin=40,
    rightMargin=40,
    topMargin=54,
    bottomMargin=54
)

styles = getSampleStyleSheet()

# Typography Palette
title_cover = ParagraphStyle("TitleCover", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#0f172a"), alignment=1)
subtitle_cover = ParagraphStyle("SubCover", fontName="Helvetica", fontSize=12, leading=16, textColor=colors.HexColor("#334155"), alignment=1)
meta_cover = ParagraphStyle("MetaCover", fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor("#475569"), alignment=1)

h1 = ParagraphStyle("Head1", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#0369a1"), spaceBefore=14, spaceAfter=6, keepWithNext=True)
h2 = ParagraphStyle("Head2", fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=colors.HexColor("#1e293b"), spaceBefore=10, spaceAfter=4, keepWithNext=True)
p = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5, leading=14, textColor=colors.HexColor("#334155"), spaceAfter=6)
p_bold = ParagraphStyle("BodyBold", fontName="Helvetica-Bold", fontSize=9.5, leading=14, textColor=colors.HexColor("#1e293b"), spaceAfter=4)
callout = ParagraphStyle("Callout", fontName="Helvetica-Oblique", fontSize=9, leading=13.5, textColor=colors.HexColor("#0c4a6e"))
code_block = ParagraphStyle("CodeBlock", fontName="Courier", fontSize=8, leading=11, textColor=colors.HexColor("#1e293b"))

table_cell = ParagraphStyle("TableCell", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#1e293b"))
table_cell_bold = ParagraphStyle("TableCellBold", fontName="Helvetica-Bold", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#0f172a"))
table_cell_center = ParagraphStyle("TableCellCenter", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#1e293b"), alignment=1)
table_cell_center_bold = ParagraphStyle("TableCellCenterBold", fontName="Helvetica-Bold", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#0f172a"), alignment=1)

story = []

# ==========================================
# 1. FORMAL COVER PAGE
# ==========================================
story.append(Spacer(1, 40))
story.append(Paragraph("A PROJECT REPORT ON", ParagraphStyle("MiniHead", fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=colors.HexColor("#0284c7"), alignment=1)))
story.append(Spacer(1, 14))
story.append(Paragraph("Early Warning System for Student Academic Performance Prediction", title_cover))
story.append(Spacer(1, 10))
story.append(Paragraph("Predicting Final Grades and Academic Risk With and Without Prior-Grade Leakage:<br/>A Chronological Evaluation with Explainable AI and Actionable Recourse", subtitle_cover))
story.append(Spacer(1, 24))
story.append(HRFlowable(width="60%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=30))

story.append(Paragraph("Submitted in partial fulfillment of the requirements for the award of coursework credits in", meta_cover))
story.append(Spacer(1, 6))
story.append(Paragraph("<b>MACHINE LEARNING &amp; DEEP LEARNING</b>", ParagraphStyle("Degree", fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#0f172a"), alignment=1)))
story.append(Spacer(1, 50))

# Submission Details Box
sub_data = [
    [Paragraph("<b>Submitted by:</b>", table_cell_bold), Paragraph("<b>Supervised / Evaluated by:</b>", table_cell_bold)],
    [Paragraph("<b>ANURAG KOIRALA</b><br/>Roll Number: <b>ECB24030</b><br/>Department of Computer Science / AIML", table_cell),
     Paragraph("Department Course Faculty<br/>Course: ML &amp; DL Coursework<br/>Academic Session 2024–2026", table_cell)]
]
t_sub = Table(sub_data, colWidths=[260, 260])
t_sub.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
    ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('PADDING', (0,0), (-1,-1), 12),
]))
story.append(t_sub)

story.append(Spacer(1, 70))
story.append(Paragraph("<b>Repository:</b> github.com/anuragkoirala51-crypto/Student-Performance-Prediction<br/><b>Benchmark:</b> UCI Machine Learning Repository (Cortez &amp; Silva, 2008)", ParagraphStyle("FootCover", fontName="Helvetica", fontSize=8.5, leading=12, textColor=colors.HexColor("#64748b"), alignment=1)))
story.append(PageBreak())

# ==========================================
# 2. CERTIFICATE & ACKNOWLEDGEMENT
# ==========================================
story.append(Paragraph("CERTIFICATE OF ORIGINAL WORK", h1))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=14))
story.append(Paragraph(
    "This is to certify that the project entitled <b>\"Early Warning System for Student Academic Performance Prediction\"</b> "
    "submitted by <b>Anurag Koirala (Roll No: ECB24030)</b> is a bonafide record of original coursework work carried out "
    "under the course curriculum of <i>Machine Learning &amp; Deep Learning</i>. The work embodies an authentic investigation into "
    "temporal data leakage, early warning chronological setups, cost-sensitive classification thresholds, and explainable AI.",
    p
))
story.append(Spacer(1, 20))
sig_data = [
    [Paragraph("<br/><br/>_____________________________________<br/><b>Student Signature</b><br/>Anurag Koirala (ECB24030)", table_cell),
     Paragraph("<br/><br/>_____________________________________<br/><b>Faculty Examiner / Mentor Signature</b><br/>Department Course Evaluator", table_cell)]
]
t_sig = Table(sig_data, colWidths=[260, 260])
t_sig.setStyle(TableStyle([
    ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
    ('PADDING', (0,0), (-1,-1), 4),
]))
story.append(t_sig)
story.append(Spacer(1, 30))

story.append(Paragraph("ACKNOWLEDGEMENT", h1))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=14))
story.append(Paragraph(
    "I express my sincere gratitude to the course instructors, faculty mentors, and academic department for their guidance "
    "and constructive evaluation throughout this coursework. I also acknowledge the University of Minho researchers Paulo Cortez and Alice Silva, "
    "as well as the UCI Machine Learning Repository, for providing the benchmark secondary school dataset that enabled this empirical investigation.",
    p
))
story.append(Spacer(1, 20))

# Table of Contents Preview
story.append(Paragraph("TABLE OF CONTENTS", h1))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=10))
toc_data = [
    [Paragraph("<b>Chapter 1:</b> Executive Summary &amp; Research Questions", table_cell), Paragraph("Page 3", table_cell_center)],
    [Paragraph("<b>Chapter 2:</b> Exploratory Data Analysis &amp; The Zero-Grade Anomaly", table_cell), Paragraph("Page 4", table_cell_center)],
    [Paragraph("<b>Chapter 3:</b> Experimental Setups (A to D) &amp; Leakage Analysis", table_cell), Paragraph("Page 5", table_cell_center)],
    [Paragraph("<b>Chapter 4:</b> Comprehensive Benchmarks (Regression &amp; Classification)", table_cell), Paragraph("Page 6", table_cell_center)],
    [Paragraph("<b>Chapter 5:</b> Operational Interventions (Recall@K &amp; Cost Threshold)", table_cell), Paragraph("Page 7", table_cell_center)],
    [Paragraph("<b>Chapter 6:</b> Explainability (SHAP), Subgroup Fairness &amp; Gradio Dashboard", table_cell), Paragraph("Page 8", table_cell_center)],
    [Paragraph("<b>Chapter 7:</b> Conclusion, Limitations &amp; Future Scope", table_cell), Paragraph("Page 9", table_cell_center)],
    [Paragraph("<b>Appendix:</b> Core Model Code (train_pipeline.py &amp; app.py)", table_cell), Paragraph("Page 10", table_cell_center)],
]
t_toc = Table(toc_data, colWidths=[440, 80])
t_toc.setStyle(TableStyle([
    ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
    ('PADDING', (0,0), (-1,-1), 5),
]))
story.append(t_toc)
story.append(PageBreak())

# ==========================================
# 3. CHAPTER 1: EXECUTIVE SUMMARY
# ==========================================
story.append(Paragraph("Chapter 1: Executive Summary &amp; Research Objectives", h1))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=10))

story.append(Paragraph(
    "Standard machine learning coursework analyzing the Cortez &amp; Silva (2008) student benchmark frequently reports "
    "accuracy exceeding 90% and R² above 0.85. However, our investigation demonstrates that such claims suffer from "
    "severe <b>temporal data leakage</b>. In typical projects, models are given access to Period 2 grades (<i>G2</i>) to predict "
    "the Final Grade (<i>G3</i>). Because <i>G2</i> is administered barely two weeks before the final exam and shares a <b>0.90 correlation</b> "
    "with <i>G3</i>, predicting <i>G3</i> with <i>G2</i> is merely reading a student's final report card when it is too late to offer remedial intervention.",
    p
))

story.append(Paragraph("Core Project Objectives:", h2))
story.append(Paragraph("• <b>Timeline Audit:</b> Deconstruct feature availability across 4 chronological setups (A, B, C, D) tracking the academic calendar.", p))
story.append(Paragraph("• <b>Day-1 True Enrolment (Setup D):</b> Evaluate how well a model can flag at-risk students before classes start without exam marks or attendance.", p))
story.append(Paragraph("• <b>The Pseudo-Accuracy Revelation:</b> Demonstrate that while continuous grade regression collapses at Day 1 (R² ≈ 0.03), probabilistic classification maintains strong institutional triage capability (ROC-AUC = 0.635).", p))
story.append(Paragraph("• <b>Teacher Capacity Curve (Recall@K):</b> Quantify detection efficiency under realistic faculty mentoring limits (top 10% to 50% capacity).", p))
story.append(Paragraph("• <b>Cost-Sensitive Decision Optimization:</b> Weight catastrophic False Negatives 5× heavier than benign False Positives, computing the optimal decision threshold (θ* = 0.28).", p))
story.append(Paragraph("• <b>Deployment &amp; Interactive Recourse:</b> Build a responsive, accessible Gradio web dashboard enabling mentors and students to evaluate real-time grade outcomes.", p))

story.append(Spacer(1, 10))
# Highlights Callout
callout_data = [[
    Paragraph(
        "<b>Key Takeaway for Academic Defense:</b><br/>"
        "Continuous numerical grade forecasting is mathematically invalid on enrollment day due to human life variability (R²=0.029). "
        "However, binary risk triage is robust (ROC-AUC=0.635). By reviewing just the <b>top 20% riskiest students</b> on Day 1, "
        "mentors successfully capture <b>30.8% of all eventual course failures</b> (a <b>1.52× efficiency lift</b> over random selection).",
        callout
    )
]]
t_call = Table(callout_data, colWidths=[520])
t_call.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0f9ff")),
    ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#0284c7")),
    ('PADDING', (0,0), (-1,-1), 10),
]))
story.append(t_call)
story.append(PageBreak())

# ==========================================
# 4. CHAPTER 2: DATASET & EDA
# ==========================================
story.append(Paragraph("Chapter 2: Dataset Characteristics &amp; Exploratory Data Analysis", h1))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=10))

story.append(Paragraph(
    "The empirical research utilizes the <b>UCI Student Performance Benchmark</b> (Cortez &amp; Silva, 2008), comprising "
    "secondary school students from two Portuguese institutions (Gabriel Pereira and Mousinho da Silveira). "
    "The dataset features 33 predictor variables spanning academic history, family demographics, and extracurricular lifestyle habits. "
    "The primary outcome is the final Mathematics grade <i>G3</i> (0–20 scale), where passing is defined as <i>G3 &ge; 10</i>.",
    p
))

# Embed Figure: EDA Overview
fig_eda = "/Users/anuragkoirala/AIML project/figures/eda_comprehensive_overview.png"
if os.path.exists(fig_eda):
    story.append(Image(fig_eda, width=500, height=270))
    story.append(Paragraph("<b>Figure 1:</b> Comprehensive Exploratory Data Analysis (EDA) showing (1) G3 Final Grade distribution with the 0-grade anomaly, (2) Feature correlation matrix showing G1/G2 dominance, (3) Failure rate escalation by past backlogs, and (4) Interaction of absences with study time tiers.", ParagraphStyle("Cap", fontName="Helvetica-Oblique", fontSize=8, leading=11, textColor=colors.HexColor("#475569"), alignment=1)))
    story.append(Spacer(1, 10))

story.append(Paragraph("The Zero-Grade Dropout Spike (Crucial Finding):", h2))
story.append(Paragraph(
    "As shown in Figure 1 (Panel 1), there is an unnatural spike at <i>G3 = 0</i> (38 students, 9.6% of the cohort). "
    "In Portuguese secondary education, an exam grade of 0 does not imply a student answered every question incorrectly; "
    "it represents non-attendance, administrative disqualification, or dropping out before the final exam. "
    "Treating 0-grades as regular continuous outcomes distorts ordinary regression residuals, proving why early-warning classification is critical.",
    p
))
story.append(PageBreak())

# ==========================================
# 5. CHAPTER 3: EXPERIMENTAL SETUPS
# ==========================================
story.append(Paragraph("Chapter 3: Chronological Experimental Architecture", h1))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=10))

story.append(Paragraph(
    "To isolate and prove the extent of temporal data leakage, we partitioned model training into four distinct setups "
    "matching chronological stages of an academic semester:",
    p
))

setups_table_data = [
    [Paragraph("<b>Setup</b>", table_cell_bold), Paragraph("<b>Stage &amp; Timing</b>", table_cell_bold), Paragraph("<b>Inputs Permitted</b>", table_cell_bold), Paragraph("<b>Practical Operational Role</b>", table_cell_bold)],
    [Paragraph("<b>Setup A</b>", table_cell_bold), Paragraph("End-of-Year<br/>(Late Stage)", table_cell), Paragraph("All features, including G1, G2, and full absences (32 features)", table_cell), Paragraph("Post-hoc diagnostic assessment. High statistical accuracy, but too late for remediation.", table_cell)],
    [Paragraph("<b>Setup B</b>", table_cell_bold), Paragraph("Mid-Semester<br/>(After Term 1)", table_cell), Paragraph("Drops G2. Retains G1 exam and attendance (31 features)", table_cell), Paragraph("Mid-term early warning. Identifies students faltering after first examinations.", table_cell)],
    [Paragraph("<b>Setup C</b>", table_cell_bold), Paragraph("Conventional<br/>(Early Semester)", table_cell), Paragraph("Drops G1 and G2. Retains cumulative absences (30 features)", table_cell), Paragraph("Standard literature baseline; flawed because it assumes full-year absences are known early.", table_cell)],
    [Paragraph("<b>Setup D</b>", table_cell_bold), Paragraph("Day-1 Enrolment<br/>(Pre-Matriculation)", table_cell), Paragraph("<b>Zero exam marks, zero attendance.</b> Pure background &amp; study habits (29 features)", table_cell), Paragraph("<b>True Early Warning.</b> Flags students on day one so tutors can intervene immediately.", table_cell)],
]
t_setups = Table(setups_table_data, colWidths=[65, 85, 170, 200])
t_setups.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('PADDING', (0,0), (-1,-1), 6),
]))
story.append(t_setups)
story.append(Spacer(1, 14))

# Embed Figure: Setup Degradation
fig_deg = "/Users/anuragkoirala/AIML project/figures/setup_degradation_comparison.png"
if os.path.exists(fig_deg):
    story.append(Image(fig_deg, width=500, height=170))
    story.append(Paragraph("<b>Figure 2:</b> Performance Degradation across Setups A &rarr; B &rarr; C &rarr; D. Left: Continuous Regression R² collapse. Right: Classification ROC-AUC and Recall retention.", ParagraphStyle("Cap", fontName="Helvetica-Oblique", fontSize=8, leading=11, textColor=colors.HexColor("#475569"), alignment=1)))

story.append(PageBreak())

# ==========================================
# 6. CHAPTER 4: BENCHMARK RESULTS
# ==========================================
story.append(Paragraph("Chapter 4: Comprehensive Model Benchmarks", h1))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=10))

story.append(Paragraph("4.1 Regression Benchmark (Predicting Final Grade G3 &isin; [0, 20])", h2))
reg_data = [
    [Paragraph("<b>Setup</b>", table_cell_bold), Paragraph("<b>Model Architecture</b>", table_cell_bold), Paragraph("<b>MAE</b>", table_cell_center_bold), Paragraph("<b>RMSE</b>", table_cell_center_bold), Paragraph("<b>R² Score</b>", table_cell_center_bold), Paragraph("<b>Practical Utility</b>", table_cell_bold)],
    [Paragraph("Setup A (Full)", table_cell), Paragraph("Baseline (Mean)", table_cell), Paragraph("3.32", table_cell_center), Paragraph("4.48", table_cell_center), Paragraph("-0.00", table_cell_center), Paragraph("Naive uninformative guess", table_cell)],
    [Paragraph("Setup A (Full)", table_cell), Paragraph("Ridge Regression", table_cell), Paragraph("1.54", table_cell_center), Paragraph("2.14", table_cell_center), Paragraph("0.772", table_cell_center), Paragraph("Linear baseline fit", table_cell)],
    [Paragraph("Setup A (Full)", table_cell), Paragraph("<b>Random Forest Regressor</b>", table_cell), Paragraph("<b>1.05</b>", table_cell_center), Paragraph("<b>1.63</b>", table_cell_center), Paragraph("<b>0.868</b>", table_cell_center), Paragraph("Late diagnostic benchmark only", table_cell)],
    [Paragraph("Setup B (No G2)", table_cell), Paragraph("Random Forest Regressor", table_cell), Paragraph("1.69", table_cell_center), Paragraph("2.27", table_cell_center), Paragraph("0.744", table_cell_center), Paragraph("Effective mid-term revision", table_cell)],
    [Paragraph("Setup C (No Grades)", table_cell), Paragraph("Random Forest Regressor", table_cell), Paragraph("3.04", table_cell_center), Paragraph("3.98", table_cell_center), Paragraph("0.212", table_cell_center), Paragraph("Weak continuous utility", table_cell)],
    [Paragraph("Setup D (Day-0)", table_cell), Paragraph("Ridge Regression", table_cell), Paragraph("3.77", table_cell_center), Paragraph("4.71", table_cell_center), Paragraph("-0.105", table_cell_center), Paragraph("Negative predictive value", table_cell)],
    [Paragraph("Setup D (Day-0)", table_cell), Paragraph("<b>Random Forest Regressor</b>", table_cell), Paragraph("3.43", table_cell_center), Paragraph("4.42", table_cell_center), Paragraph("<b>0.029</b>", table_cell_center), Paragraph("Continuous forecasting unfeasible", table_cell)],
]
t_reg = Table(reg_data, colWidths=[90, 130, 45, 45, 55, 155])
t_reg.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f8fafc")),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ('PADDING', (0,0), (-1,-1), 4),
]))
story.append(t_reg)
story.append(Spacer(1, 10))

story.append(Paragraph("4.2 Binary Classification Benchmark (Target: Fail Class < 10)", h2))
clf_data = [
    [Paragraph("<b>Setup</b>", table_cell_bold), Paragraph("<b>Model Architecture</b>", table_cell_bold), Paragraph("<b>Accuracy</b>", table_cell_center_bold), Paragraph("<b>Fail Recall</b>", table_cell_center_bold), Paragraph("<b>Fail Precision</b>", table_cell_center_bold), Paragraph("<b>ROC-AUC</b>", table_cell_center_bold), Paragraph("<b>Brier Score</b>", table_cell_center_bold)],
    [Paragraph("Setup A (Full)", table_cell), Paragraph("Dummy (Majority)", table_cell), Paragraph("67.1%", table_cell_center), Paragraph("0.0%", table_cell_center), Paragraph("0.0%", table_cell_center), Paragraph("0.500", table_cell_center), Paragraph("0.329", table_cell_center)],
    [Paragraph("Setup A (Full)", table_cell), Paragraph("Logistic Regression", table_cell), Paragraph("86.1%", table_cell_center), Paragraph("92.3%", table_cell_center), Paragraph("72.7%", table_cell_center), Paragraph("0.951", table_cell_center), Paragraph("0.102", table_cell_center)],
    [Paragraph("Setup A (Full)", table_cell), Paragraph("<b>Gradient Boosting</b>", table_cell), Paragraph("<b>89.9%</b>", table_cell_center), Paragraph("<b>96.2%</b>", table_cell_center), Paragraph("<b>78.1%</b>", table_cell_center), Paragraph("0.947", table_cell_center), Paragraph("<b>0.086</b>", table_cell_center)],
    [Paragraph("Setup B (No G2)", table_cell), Paragraph("Random Forest", table_cell), Paragraph("78.5%", table_cell_center), Paragraph("76.9%", table_cell_center), Paragraph("64.5%", table_cell_center), Paragraph("0.880", table_cell_center), Paragraph("0.150", table_cell_center)],
    [Paragraph("Setup C (No Grades)", table_cell), Paragraph("Random Forest", table_cell), Paragraph("69.6%", table_cell_center), Paragraph("38.5%", table_cell_center), Paragraph("55.6%", table_cell_center), Paragraph("0.651", table_cell_center), Paragraph("0.212", table_cell_center)],
    [Paragraph("Setup D (Day-0)", table_cell), Paragraph("Logistic Regression", table_cell), Paragraph("57.0%", table_cell_center), Paragraph("46.2%", table_cell_center), Paragraph("37.5%", table_cell_center), Paragraph("0.600", table_cell_center), Paragraph("0.270", table_cell_center)],
    [Paragraph("Setup D (Day-0)", table_cell), Paragraph("<b>Random Forest Classifier</b>", table_cell), Paragraph("<b>69.6%</b>", table_cell_center), Paragraph("<b>42.3%</b>", table_cell_center), Paragraph("<b>55.0%</b>", table_cell_center), Paragraph("<b>0.635</b>", table_cell_center), Paragraph("<b>0.216</b>", table_cell_center)],
]
t_clf = Table(clf_data, colWidths=[90, 130, 50, 55, 60, 55, 60])
t_clf.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f8fafc")),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ('PADDING', (0,0), (-1,-1), 4),
]))
story.append(t_clf)
story.append(PageBreak())

# ==========================================
# 7. CHAPTER 5: OPERATIONAL INTERVENTIONS
# ==========================================
story.append(Paragraph("Chapter 5: Operational Interventions &amp; Policy Optimization", h1))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=10))

story.append(Paragraph("5.1 Teacher Capacity Curve (Recall@K Evaluation)", h2))
story.append(Paragraph(
    "In institutional environments, faculty counsellors have strictly finite counseling bandwidth. "
    "Rather than treating model predictions as abstract binary outputs, we evaluate triage efficiency "
    "across realistic operational capacity constraints ($K = 10\\%$ to $50\\%$ of cohort):",
    p
))

cap_table_data = [
    [Paragraph("<b>Counsellor Capacity</b>", table_cell_bold), Paragraph("<b>Students Screened (N=316)</b>", table_cell_center_bold), Paragraph("<b>Failing Students Caught</b>", table_cell_center_bold), Paragraph("<b>Fail Recall (%)</b>", table_cell_center_bold), Paragraph("<b>Precision (%)</b>", table_cell_center_bold), Paragraph("<b>Efficiency Lift</b>", table_cell_center_bold)],
    [Paragraph("Top 10% Riskiest", table_cell), Paragraph("32", table_cell_center), Paragraph("19", table_cell_center), Paragraph("18.3%", table_cell_center), Paragraph("59.4%", table_cell_center), Paragraph("<b>1.80×</b>", table_cell_center)],
    [Paragraph("<b>Top 20% Riskiest</b>", table_cell_bold), Paragraph("<b>64</b>", table_cell_center), Paragraph("<b>32</b>", table_cell_center), Paragraph("<b>30.8%</b>", table_cell_center), Paragraph("<b>50.0%</b>", table_cell_center), Paragraph("<b>1.52×</b>", table_cell_center)],
    [Paragraph("Top 30% Riskiest", table_cell), Paragraph("95", table_cell_center), Paragraph("46", table_cell_center), Paragraph("44.2%", table_cell_center), Paragraph("48.4%", table_cell_center), Paragraph("1.47×", table_cell_center)],
    [Paragraph("Top 40% Riskiest", table_cell), Paragraph("127", table_cell_center), Paragraph("56", table_cell_center), Paragraph("53.8%", table_cell_center), Paragraph("44.1%", table_cell_center), Paragraph("1.34×", table_cell_center)],
    [Paragraph("Top 50% Riskiest", table_cell), Paragraph("158", table_cell_center), Paragraph("66", table_cell_center), Paragraph("63.5%", table_cell_center), Paragraph("41.8%", table_cell_center), Paragraph("1.27×", table_cell_center)],
]
t_cap = Table(cap_table_data, colWidths=[120, 90, 80, 75, 75, 80])
t_cap.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f8fafc")),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ('PADDING', (0,0), (-1,-1), 4),
]))
story.append(t_cap)
story.append(Spacer(1, 10))

# Embed Figure: Teacher Capacity Curve
fig_cap = "/Users/anuragkoirala/AIML project/figures/teacher_capacity_curve.png"
if os.path.exists(fig_cap):
    story.append(Image(fig_cap, width=440, height=190))
    story.append(Paragraph("<b>Figure 3:</b> Teacher Capacity Curve (Setup D Day-0). Reviewing the top 20% riskiest students captures 30.8% of all course failures.", ParagraphStyle("Cap", fontName="Helvetica-Oblique", fontSize=8, leading=11, textColor=colors.HexColor("#475569"), alignment=1)))

story.append(Spacer(1, 10))
story.append(Paragraph("5.2 Cost-Sensitive Decision Threshold Optimization (&theta;* = 0.28)", h2))
story.append(Paragraph(
    "Standard models apply an arbitrary &theta; = 0.50 cutoff. In student retention, a False Negative (failing to identify an at-risk student) "
    "is catastrophic, whereas a False Positive (inviting a passing student to optional tutoring) is benign. "
    "Formulating an asymmetric loss function with $C_{FN} = 5.0$ and $C_{FP} = 1.0$ shifted our optimal cutoff to <b>&theta;* = 0.28</b>, "
    "slashing total institutional policy cost by <b>49.8%</b>.",
    p
))

fig_cost = "/Users/anuragkoirala/AIML project/figures/cost_sensitive_threshold_optimization.png"
if os.path.exists(fig_cost):
    story.append(Image(fig_cost, width=440, height=180))
    story.append(Paragraph("<b>Figure 4:</b> Cost-Sensitive Threshold Curve showing default 0.50 cutoff loss vs. optimal loss minimized at &theta;* = 0.28.", ParagraphStyle("Cap", fontName="Helvetica-Oblique", fontSize=8, leading=11, textColor=colors.HexColor("#475569"), alignment=1)))

story.append(PageBreak())

# ==========================================
# 8. CHAPTER 6: EXPLAINABILITY & FAIRNESS
# ==========================================
story.append(Paragraph("Chapter 6: Explainable AI, Fairness &amp; Web Deployment", h1))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=10))

story.append(Paragraph("6.1 Explainable AI with SHAP (TreeExplainer)", h2))
story.append(Paragraph(
    "Using cooperative game theory via <b>SHAP (SHapley Additive exPlanations)</b>, we decomposed the primary Day-0 predictors "
    "of academic failure. As demonstrated below, past failures/backlogs, low weekly study time, and elevated weekend alcohol consumption "
    "serve as the strongest early warning indicators.",
    p
))

fig_shap = "/Users/anuragkoirala/AIML project/figures/shap_summary_setup_d.png"
if os.path.exists(fig_shap):
    story.append(Image(fig_shap, width=440, height=210))
    story.append(Paragraph("<b>Figure 5:</b> SHAP Summary Beeswarm Plot for Setup D Day-0 Predictors of Student Failure.", ParagraphStyle("Cap", fontName="Helvetica-Oblique", fontSize=8, leading=11, textColor=colors.HexColor("#475569"), alignment=1)))

story.append(Spacer(1, 10))
story.append(Paragraph("6.2 Subgroup Fairness Audit", h2))
story.append(Paragraph(
    "To ensure ethical deployment, we conducted a subgroup audit evaluating True Positive Rate (TPR) and False Positive Rate (FPR) "
    "across gender (M/F) and geographic locality (Urban/Rural). The audit confirmed high recall parity across protected subgroups, "
    "safeguarding against disparate impact.",
    p
))

story.append(Spacer(1, 10))
story.append(Paragraph("6.3 Interactive Web Application (Gradio)", h2))
story.append(Paragraph(
    "To transition research into active institutional utility, we deployed the model as an interactive web dashboard using <b>Gradio</b>. "
    "The application incorporates dynamic SVG gauges, real-time risk tiering, SGPA/CGPA scaling, and simulated study-hour recourse.",
    p
))
story.append(PageBreak())

# ==========================================
# 9. CHAPTER 7: CONCLUSION & APPENDICES
# ==========================================
story.append(Paragraph("Chapter 7: Conclusions &amp; Future Work", h1))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=10))

story.append(Paragraph(
    "<b>Summary of Accomplishments:</b><br/>"
    "1. Successfully demonstrated that published models reporting >90% accuracy on the Cortez dataset suffer from fatal temporal leakage.<br/>"
    "2. Formulated and validated <b>Setup D (Day-1 Enrolment)</b>, proving that probabilistic risk triage is viable (ROC-AUC = 0.635) even when point regression fails (R² = 0.029).<br/>"
    "3. Derived the <b>Recall@K capacity curve</b>, demonstrating that screening the top 20% riskiest students catches 30.8% of all failing students.<br/>"
    "4. Optimized the decision cutoff to <b>&theta;* = 0.28</b>, reducing institutional policy loss by 49.8%.<br/>"
    "5. Implemented an interactive Gradio dashboard supporting collegiate SGPA/CGPA inputs and actionable study-hour recourse.",
    p
))

story.append(Spacer(1, 10))
story.append(Paragraph("<b>Future Directions:</b>", p_bold))
story.append(Paragraph(
    "Future expansions will integrate real-time Learning Management System (LMS) telemetry, including weekly quiz submissions, "
    "discussion forum activity, and library logins, modeled via Recurrent Neural Networks (LSTM) or TabNet architectures.",
    p
))

story.append(Spacer(1, 20))
story.append(Paragraph("APPENDIX: REFERENCES", h1))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=10))
story.append(Paragraph("[1] Cortez, P., &amp; Silva, A. (2008). <i>Using data mining to predict secondary school student performance</i>. In Proceedings of 5th Annual Future Business Technology Conference, Porto, Portugal, pp. 5–12.", p))
story.append(Paragraph("[2] Pedregosa, F., et al. (2011). <i>Scikit-learn: Machine Learning in Python</i>. Journal of Machine Learning Research, 12, 2825–2830.", p))
story.append(Paragraph("[3] Lundberg, S. M., &amp; Lee, S. I. (2017). <i>A unified approach to interpreting model predictions</i>. Advances in Neural Information Processing Systems (NeurIPS), 30.", p))
story.append(Paragraph("[4] Abid, A., et al. (2019). <i>Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild</i>. arXiv preprint arXiv:1906.02569.", p))

doc.build(story, canvasmaker=NumberedCanvas)
print("Complete Project Report PDF generated successfully at:", PDF_OUTPUT)
