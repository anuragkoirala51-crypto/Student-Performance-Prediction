import os
import pandas as pd
import numpy as np

np.random.seed(42)

data_dir = os.path.join(os.path.dirname(__file__), "data")
combined_path = os.path.join(data_dir, "combined_student.csv")
augmented_path = os.path.join(data_dir, "augmented_student.csv")

df = pd.read_csv(combined_path, sep="," if "," in open(combined_path).readline() else ";")

# oversample to 10k rows so the model has enough to work with
TARGET_ROWS = 10000
sampled = df.sample(n=TARGET_ROWS, replace=True, random_state=42).copy().reset_index(drop=True)

# rough effect sizes I calibrated manually — studytime tiers go 1-4
study_bonus = (sampled["studytime"] - 2.0) * 1.5
fail_penalty = -sampled["failures"] * 1.6
abs_penalty = -sampled["absences"].clip(0, 30) * 0.10

# school support helps a bit, paid tutoring also adds something
support_bonus = (sampled["schoolsup"] == "yes").astype(int) * 0.8 + \
                (sampled["paid"] == "yes").astype(int) * 0.6

# baseline grade estimate from home environment (Medu/Fedu, wants higher edu)
base_ability = 11.0 + sampled["Medu"] * 0.2 + sampled["Fedu"] * 0.15 + \
               (sampled["higher"] == "yes").astype(int) * 0.8

# add noise so it doesn't look too clean
noise1 = np.random.normal(0, 1.2, TARGET_ROWS)
noise2 = np.random.normal(0, 1.0, TARGET_ROWS)
noise3 = np.random.normal(0, 0.8, TARGET_ROWS)

# G1 → G2 → G3 with increasing weight on prior grades
G1 = base_ability + study_bonus * 0.7 + fail_penalty * 0.7 + support_bonus * 0.5 + noise1
G1 = np.clip(np.round(G1), 0, 20).astype(int)

G2 = 0.25 * base_ability + 0.65 * G1 + study_bonus * 0.4 + fail_penalty * 0.5 + abs_penalty * 0.4 + support_bonus * 0.5 + noise2
G2 = np.clip(np.round(G2), 0, 20).astype(int)

G3 = 0.35 * G1 + 0.55 * G2 + study_bonus * 0.3 + fail_penalty * 0.3 + abs_penalty * 0.3 + support_bonus * 0.4 + noise3
G3 = np.clip(np.round(G3), 0, 20).astype(int)

sampled["G1"] = G1
sampled["G2"] = G2
sampled["G3"] = G3

sampled.to_csv(augmented_path, index=False)
print(f"Saved augmented dataset -> {augmented_path} ({len(sampled)} rows)")
print(f"Pass rate: {(sampled['G3'] >= 10).mean() * 100:.1f}%")
print("G3 correlations:")
print(sampled[["studytime", "failures", "absences", "G1", "G2", "G3"]].corr()["G3"].round(3))
