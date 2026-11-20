import os
import pandas as pd
import numpy as np

# Set reproducibility seed
np.random.seed(42)

# Paths
data_dir = os.path.join(os.path.dirname(__file__), "data")
combined_path = os.path.join(data_dir, "combined_student.csv")
augmented_path = os.path.join(data_dir, "augmented_student.csv")

# Load base combined dataset
df = pd.read_csv(combined_path, sep="," if "," in open(combined_path).readline() else ";")

TARGET_ROWS = 10000
sampled = df.sample(n=TARGET_ROWS, replace=True, random_state=42).copy().reset_index(drop=True)

# Feature effect calibrations
# 1. Study time effect (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)
study_bonus = (sampled["studytime"] - 2.0) * 1.5

# 2. Backlog / prior failures penalty
fail_penalty = -sampled["failures"] * 1.6

# 3. Absence penalty
abs_penalty = -sampled["absences"].clip(0, 30) * 0.10

# 4. Support bonus (Tutoring and Academic Support help students improve)
support_bonus = (sampled["schoolsup"] == "yes").astype(int) * 0.8 + (sampled["paid"] == "yes").astype(int) * 0.6

# 5. Baseline ability from academic environment
base_ability = 11.0 + sampled["Medu"] * 0.2 + sampled["Fedu"] * 0.15 + (sampled["higher"] == "yes").astype(int) * 0.8

# Add small natural variance / noise
noise1 = np.random.normal(0, 1.2, TARGET_ROWS)
noise2 = np.random.normal(0, 1.0, TARGET_ROWS)
noise3 = np.random.normal(0, 0.8, TARGET_ROWS)

# Generate coherent internal and final exam scores on 0-20 scale
G1 = base_ability + study_bonus * 0.7 + fail_penalty * 0.7 + support_bonus * 0.5 + noise1
G1 = np.clip(np.round(G1), 0, 20).astype(int)

G2 = 0.25 * base_ability + 0.65 * G1 + study_bonus * 0.4 + fail_penalty * 0.5 + abs_penalty * 0.4 + support_bonus * 0.5 + noise2
G2 = np.clip(np.round(G2), 0, 20).astype(int)

G3 = 0.35 * G1 + 0.55 * G2 + study_bonus * 0.3 + fail_penalty * 0.3 + abs_penalty * 0.3 + support_bonus * 0.4 + noise3
G3 = np.clip(np.round(G3), 0, 20).astype(int)

sampled["G1"] = G1
sampled["G2"] = G2
sampled["G3"] = G3

# Save augmented dataset
sampled.to_csv(augmented_path, index=False)
print(f"Realistic augmented dataset saved to {augmented_path} with {len(sampled)} rows.")
print(f"Overall Pass Rate: {(sampled['G3'] >= 10).mean() * 100:.1f}%")
print("Correlations with G3:")
print(sampled[["studytime", "failures", "absences", "G1", "G2", "G3"]].corr()["G3"].round(3))
