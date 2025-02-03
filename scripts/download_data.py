import os
import kaggle

DATASET_NAME = "pavansubhasht/ibm-hr-analytics-attrition-dataset"

DATA_DIR = os.path.join("data", "raw")

os.makedirs(DATA_DIR, exist_ok=True)

# Make sure to have your Kaggle API key saved in your home directory
# as ~/.kaggle/kaggle.json or C:\Users\<username>\.kaggle\kaggle.json
print(f"Downloading dataset: `{DATASET_NAME}` ...")
kaggle.api.dataset_download_files(DATASET_NAME, path=DATA_DIR, unzip=True)

print("✅ Dataset downloaded and extracted successfully.")

# Renaming the dataset to `hr_employee_attrition.csv`
DATA_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv") 

os.rename(DATA_PATH, os.path.join(DATA_DIR, "hr_employee_attrition.csv"))

print("✅ Dataset renamed successfully to `hr_employee_attrition.csv`.")