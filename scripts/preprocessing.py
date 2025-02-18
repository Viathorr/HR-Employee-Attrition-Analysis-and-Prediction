import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from feature_engineering import FeatureEngineeringTransformer, FeatureRemover

DATA_PATH = os.path.join( "data", "raw", "hr_employee_attrition.csv")
SAVE_PATH = os.path.join("data", "processed", "hr_employee_attrition.csv")

df = pd.read_csv(DATA_PATH)

def load_data(data_path):
    return pd.read_csv(data_path)

def save_data(df, save_path):
    df.to_csv(save_path, index=False)

def get_preprocessing_pipeline(target_col, ord_cat_cols, ord_categories, nom_cat_cols, features_to_remove):
    column_transformer = ColumnTransformer([
        ("ord_encoder", OrdinalEncoder(categories=ord_categories), ord_cat_cols),
        ("onehot_encoder", OneHotEncoder(sparse_output=False, drop="first"), nom_cat_cols)
    ],
        verbose_feature_names_out=False,
        remainder="passthrough"
    )

    preprocessing_pipeline = Pipeline([
        ("feature_engineering", FeatureEngineeringTransformer(log_transform=True, winsorize_transform=True)),
        ("feature_remover", FeatureRemover(features_to_remove)),
        ("cols_transformer", column_transformer),        
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler())
    ])

    return preprocessing_pipeline

# Load data
df = load_data(DATA_PATH)

# Define parameters
target_col = "Attrition"
features_to_remove = ["EmployeeCount", "StandardHours", "PerformanceRating", "Education", "EmployeeNumber", "Over18", "PercentSalaryHike", "MonthlyIncome"]

ord_cat_cols = ["OverTime", "BusinessTravel"]
ord_categories = [
    ["Yes", "No"],
    ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
]
nom_cat_cols = ["Department", "EducationField", "Gender", "JobRole", "MaritalStatus"]

# Preprocessing
preprocessing_pipeline = get_preprocessing_pipeline(target_col, ord_cat_cols, ord_categories, nom_cat_cols, features_to_remove)

X, y = df.drop(columns=[target_col]), df[target_col]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

df_transformed = preprocessing_pipeline.fit_transform(X)

column_names = preprocessing_pipeline.named_steps["cols_transformer"].get_feature_names_out()

df_transformed = pd.DataFrame(df_transformed, columns=column_names)
df_transformed[target_col] = y

save_data(df_transformed, SAVE_PATH)

joblib.dump(preprocessing_pipeline, os.path.join("models", "preprocessing_pipeline.joblib"))