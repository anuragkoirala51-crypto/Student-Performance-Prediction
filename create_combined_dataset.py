import pandas as pd
import os

# Paths to existing datasets
data_dir = os.path.join(os.path.dirname(__file__), "data")
mat_path = os.path.join(data_dir, "student-mat.csv")
por_path = os.path.join(data_dir, "student-por.csv")
combined_path = os.path.join(data_dir, "combined_student.csv")

# Load both CSVs (they have the same columns)
mat_df = pd.read_csv(mat_path, sep=";")
por_df = pd.read_csv(por_path, sep=";")

# Concatenate them
combined_df = pd.concat([mat_df, por_df], ignore_index=True)

# Save combined dataset
combined_df.to_csv(combined_path, index=False)
print(f"Combined dataset saved to {combined_path} with {len(combined_df)} rows")
