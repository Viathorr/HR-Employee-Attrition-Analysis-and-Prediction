import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

FIGS_DIR = os.path.abspath(os.path.join("..", "reports", "figures"))

def save_figure(fig: matplotlib.figure.Figure, filename: str, dirname: str):
  """
  Saves a given figure to a specified folder.

  Args:
    fig (matplotlib.figure.Figure): Figure object to save.
    filename (str): Name of the file (without extension).
    dirname (str): Name of the directory to save the figure in. 

  Returns:
    None
  """
  folder = os.path.join(FIGS_DIR, dirname)
  
  os.makedirs(folder, exist_ok=True)
  save_path = os.path.join(folder, f"{filename}.png")
  
  fig.savefig(save_path, bbox_inches='tight')
  
  print(f"✅ Figure saved.")


def cat_distribution(df, col_name, ratio=False) -> matplotlib.figure.Figure:
  """
  Visualizes the distribution of a categorical column in a given DataFrame.

  Args:  
    df (pandas.DataFrame): DataFrame containing the column to visualize.
    col_name (str): Name of the column to visualize.
    ratio (bool, optional): If True, displays the ratio of unique values in the column. Default is False.

  Returns:
    matplotlib.figure.Figure: Figure object containing the distribution plot.
  """
  if col_name not in df.columns:
    raise ValueError(f"Column '{col_name}' not found in DataFrame.")
  
  if ratio:
    print(f"\n\n📌Ratio of unique values in '{col_name}':")
    print(pd.DataFrame(df[col_name].value_counts(normalize=True) * 100).rename(columns={"proportion": "Ratio (%)"}).sort_values("Ratio (%)", ascending=False))
    print("-" * 50)
    
  fig = plt.figure(figsize=(10, 5))
  
  sns.countplot(data=df, x=col_name, hue=col_name, palette="Set3", legend=False, order=df[col_name].value_counts().index)
  plt.title(f"Distribution of '{col_name}'")
  plt.xlabel(f"'{col_name}'")
  plt.ylabel("Count")
  
  plt.show()
  
  return fig
  
  
def num_distribution(df, col_name):
  """
  Visualizes the distribution of a numerical column in a given DataFrame.

  Args:
    df (pandas.DataFrame): DataFrame containing the column to visualize.
    col_name (str): Name of the column to visualize.

  Returns:
    matplotlib.figure.Figure: Figure object containing the distribution plot.
  """
  if col_name not in df.columns:
    raise ValueError(f"Column '{col_name}' not found in DataFrame.")
  
