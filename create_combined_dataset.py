import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(__file__), "data")
mat_path = os.path.join(data_dir, "student-mat.csv")
por_path = os.path.join(data_dir, "student-por.csv")
combined_path = os.path.join(data_dir, "combined_student.csv")

# both files have the same columns, just different subjects (math vs portuguese)
mat_df = pd.read_csv(mat_path, sep=";")
por_df = pd.read_csv(por_path, sep=";")

combined_df = pd.concat([mat_df, por_df], ignore_index=True)

combined_df.to_csv(combined_path, index=False)
print(f"Combined dataset saved -> {combined_path} ({len(combined_df)} rows)")
