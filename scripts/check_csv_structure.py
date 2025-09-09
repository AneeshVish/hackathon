import pandas as pd
import os

def check_csv_structure():
    data_dir = "data/raw"
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            try:
                filepath = os.path.join(data_dir, filename)
                df = pd.read_csv(filepath, nrows=1)
                print(f"\nFile: {filename}")
                print("Columns:", df.columns.tolist())
                print("First row:", df.iloc[0].to_dict())
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")

if __name__ == "__main__":
    check_csv_structure()
