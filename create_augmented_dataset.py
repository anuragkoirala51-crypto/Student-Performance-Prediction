import os
import pandas as pd
import numpy as np

# Paths
data_dir = os.path.join(os.path.dirname(__file__), "data")
combined_path = os.path.join(data_dir, "combined_student.csv")
augmented_path = os.path.join(data_dir, "augmented_student.csv")

# Load combined dataset
df = pd.read_csv(combined_path, sep="," if "," in open(combined_path).readline() else ";")

# Number of extra rows to add (e.g., make total ~3000 rows)
TARGET_ROWS = 3000
extra = TARGET_ROWS - len(df)
if extra > 0:
    # Sample with replacement and add small noise to numeric columns
    sampled = df.sample(n=extra, replace=True, random_state=42).copy()
    # Add slight random jitter to numeric columns (age, failures, etc.)
    numeric_cols = sampled.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # add noise within +/-1 where reasonable
        sampled[col] = sampled[col] + np.random.randint(-1, 2, size=len(sampled))
        # keep values within original min/max
        col_min, col_max = df[col].min(), df[col].max()
        sampled[col] = sampled[col].clip(col_min, col_max)
    df_aug = pd.concat([df, sampled], ignore_index=True)
else:
    df_aug = df

# Save augmented dataset
df_aug.to_csv(augmented_path, index=False)
print(f"Augmented dataset saved to {augmented_path} with {len(df_aug)} rows (added {max(0, extra)} rows)")
