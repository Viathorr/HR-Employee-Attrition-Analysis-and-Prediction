import os
import numpy as np
import pandas as pd


def get_cols_names(df: pd.DataFrame, cat_threshold: int = 10, car_threshold: int = 20):
  """
  Identify and separate categorical and numerical columns of a given DataFrame.

  Args:
    df (pd.DataFrame): Input DataFrame.
    cat_threshold (int, optional): Threshold for considering a column as categorical. Default is 10.
    car_threshold (int, optional): Threshold for considering a column as cardinal. Default is 20.

  Returns:
    tuple: A tuple containing the following lists:
      cat_cols (list): List of categorical columns.
      num_cols (list): List of numerical columns.
      num_but_cat_cols (list): List of numerical columns that are categorical.
      cat_but_car_cols (list): List of categorical columns that are cardinal.
  """
  
  cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
  num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
  cat_but_car_cols = [col for col in cat_cols if df[col].nunique() > car_threshold]
  num_but_cat_cols = [col for col in num_cols if df[col].nunique() <= cat_threshold]

  # Combining categorical features and excluding categorical but cardinal features
  cat_cols += num_but_cat_cols
  cat_cols = [col for col in cat_cols if col not in cat_but_car_cols]

  # Exluding numeric-looking categorical features from numeric
  num_cols = [col for col in num_cols if col not in num_but_cat_cols]

  return cat_cols, num_cols, num_but_cat_cols, cat_but_car_cols


def print_cols_summary(df: pd.DataFrame, cat_cols, num_cols, num_but_cat_cols=None, cat_but_car_cols=None):
    """
    Prints a summary of the given DataFrame's columns.

    Args:
      df (pd.DataFrame): Input DataFrame.
      cat_cols (list): List of categorical columns.
      num_cols (list): List of numerical columns.
      num_but_cat_cols (list, optional): List of numerical columns that are categorical. Default is None.
      cat_but_car_cols (list, optional): List of categorical columns that are cardinal. Default is None.  
      
    Returns:
      None: Prints the summary of the DataFrame's columns.
    """
    print(f"Categorical columns ({len(cat_cols)}) (including numeric-looking categorical columns): {cat_cols}\n\nFrom them numeric-looking categorical columns ({len(num_but_cat_cols)}): {num_but_cat_cols}\n")
    print("-" * 50)
    print(f"\nNumeric columns ({len(num_cols)}): {num_cols}\n")
    print("-" * 50)
    print(f"\nCategorical but cardinal columns ({len(cat_but_car_cols)}): {cat_but_car_cols}\n")
    print("-" * 50)

    for col in cat_cols:
      print(f"\n'{col}' column contains {df[col].nunique()} unique values: {df[col].unique().tolist()}")

    print("\n", "-" * 50)
    
    for col in num_cols:
      print(f"\n'{col}' column contains values in a range from {df.describe().loc['min', col]} to {df.describe().loc['max', col]}")

    print("\n", "-" * 50)
    for col in cat_but_car_cols:
      print(f"\n'{col}' column contains {df[col].nunique()} unique values.")
        
        
        
        
