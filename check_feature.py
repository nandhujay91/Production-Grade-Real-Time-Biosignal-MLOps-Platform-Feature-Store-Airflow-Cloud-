import pandas as pd

file_path = "data/features/session_20260329_192631/features_20260329_192659.parquet"

df = pd.read_parquet(file_path)

print(df.head())
print(df.columns)
print(df.describe())