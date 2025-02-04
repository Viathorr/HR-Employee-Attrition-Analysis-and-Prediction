import os
import numpy as np
import pandas as pd


def get_cols_names(df: pd.DataFrame, cat_threshold: int = 10, car_threshold: int = 20):
  """
  Identify and separate categorical and numerical columns of a given DataFrame.

  Parameters
  ----------
  df : pd.DataFrame
    Input DataFrame.
  cat_threshold : int, optional
    Number of unique values below which a column is considered categorical, by default 10.
  car_threshold : int, optional
    Number of unique values above which a column is considered categorical but cardinal, by default 20.

  Returns
  -------
  tuple
    A tuple containing the categorical columns, numerical columns,
    numerical columns that are categorical, and categorical columns that are cardinal.
  """
  
  cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
  num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
  cat_but_car_cols = [col for col in cat_cols if df[col].nunique() > car_threshold]
  num_but_cat_cols = [col for col in num_cols if df[col].nunique() < cat_threshold]

  # Combining categorical features and excluding categorical but cardinal features
  cat_cols += num_but_cat_cols
  cat_cols = [col for col in cat_cols if col not in cat_but_car_cols]

  # Exluding numeric-looking categorical features from numeric
  num_cols = [col for col in num_cols if col not in num_but_cat_cols]

  return cat_cols, num_cols, num_but_cat_cols, cat_but_car_cols


def print_cols_summary(df: pd.DataFrame, cat_cols, num_cols, num_but_cat_cols=None, cat_but_car_cols=None):
    """
    Prints a summary of the given DataFrame's columns.

    Parameters
    ----------
    cat_cols : list
        List of categorical columns.
    num_cols : list
        List of numerical columns.
    num_but_cat_cols : list, optional
        List of numeric-looking categorical columns, by default None.
    cat_but_car_cols : list, optional
        List of categorical but cardinal columns, by default None.
    """
    print(f"Categorical columns (including numeric-looking categorical columns): {cat_cols}\n\nFrom them numeric-looking categorical columns: {num_but_cat_cols}\n")
    print("-" * 50)
    print(f"\nNumeric columns: {num_cols}\n")
    print("-" * 50)
    print(f"\nCategorical but cardinal columns: {cat_but_car_cols}\n")
    print("-" * 50)

    for col in cat_cols:
        print(f"\n'{col}' column contains {df[col].nunique()} unique values: {df[col].unique().tolist()}")

    print("\n", "-" * 50)
    
    for col in num_cols:
        print(f"\n'{col}' column contains values in a range from {df.describe().loc['min', col]} to {df.describe().loc['max', col]}")

    print("\n", "-" * 50)
    for col in cat_but_car_cols:
        print(f"\n'{col}' column contains {df[col].nunique()} unique values.")