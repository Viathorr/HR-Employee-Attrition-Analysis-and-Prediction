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

# Distributions Visualisation

def cat_distribution(df: pd.DataFrame, col_name: str, ax: plt.Axes, ratio: bool = False) -> None:
  """
  Visualizes the distribution of a categorical column in a given DataFrame.

  Args:  
    df (pandas.DataFrame): DataFrame containing the column to visualize.
    col_name (str): Name of the column to visualize.
    ax (matplotlib.axes.Axes): Axes object to draw the plot.
    ratio (bool, optional): If True, displays the ratio of unique values in the column. Default is False.

  Returns:
    None: Plot is drawn directly on the provided axes object.
  """
  if col_name not in df.columns:
    raise ValueError(f"Column '{col_name}' not found in DataFrame.")
  
  if ratio:
    print(f"\n\n📌Ratio of unique values in '{col_name}':")
    print(pd.DataFrame(df[col_name].value_counts(normalize=True) * 100).rename(columns={"proportion": "Ratio (%)"}).sort_values("Ratio (%)", ascending=False))
    print("-" * 50)
  
  sns.countplot(data=df, x=col_name, hue=col_name, palette="Set3", legend=False, order=df[col_name].value_counts().index, ax=ax)
  ax.set_title(f"Distribution of '{col_name}'")
  ax.set_xlabel(f"'{col_name}'")
  ax.set_ylabel("Count")
  
  if df[col_name].nunique() > 5:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
  
  
def num_distribution(df: pd.DataFrame, col_name: str):
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
  
  # TBA
  

# Feature-Target Relationships

def target_by_cat(df: pd.DataFrame, target_col: str, col_name: str, ax: plt.Axes):
  """
  Visualizes the target distribution by a categories in a given column in a given DataFrame.

  Args:
    df (pandas.DataFrame): DataFrame containing the columns to visualize.
    target_col (str): Name of the target column.
    col_name (str): Name of the categorical column.
    ax (matplotlib.axes.Axes): Axes object to draw the plot.

  Returns:
    None: Plot is drawn directly on the provided axes object.
  """
  if col_name not in df.columns:
    raise ValueError(f"Column '{col_name}' not found in DataFrame.")
  
  if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found in DataFrame.")
  
  cross_tab = pd.crosstab(df[col_name], df[target_col], normalize="index") * 100
  bar_plot = cross_tab.plot(kind="bar", stacked=True, colormap="coolwarm", ax=ax)
  
  ax.set_title(f"Target ({target_col}) Percentage Distribution by {col_name}")
  ax.set_xlabel(f"'{col_name}'")
  ax.set_ylabel("Target Percentage (%)")
  ax.legend(loc="lower center", bbox_to_anchor=(1.1, 0.1))
  
  if df[col_name].nunique() > 5:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
  else:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
  
  for container in bar_plot.containers:
    bar_plot.bar_label(container, fmt="%.1f%%", label_type="center")


def target_by_num(df: pd.DataFrame, target_col: str, col_name: str):
  """
  Visualizes the target distribution by a numerical column in a given DataFrame.

  Args:
    df (pandas.DataFrame): DataFrame containing the columns to visualize.
    target_col (str): Name of the target column.
    col_name (str): Name of the numerical column.

  Returns:
    matplotlib.figure.Figure: Figure object containing the relationship plot.
  """
  if col_name not in df.columns:
    raise ValueError(f"Column '{col_name}' not found in DataFrame.")
  
  if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found in DataFrame.")
  
  fig = plt.figure(figsize=(10, 5))
  
  sns.stripplot(data=df, x=target_col, y=col_name, palette="Set3")
  
  plt.title(f"Target ({target_col}) Distribution by {col_name}.")
  plt.xlabel(f"{target_col}")
  plt.ylabel(f"{col_name}")
  
  plt.show()
  
  return fig

  
def plot_dual_distributions(df: pd.DataFrame, target_col: str, col_name: str, ratio: bool = True) -> None:
  """
  Combines the 'target_by_cat' and 'cat_distribution' plots into a single figure with two subplots.

  Args:
    df (pandas.DataFrame): DataFrame containing the columns to visualize.
    target_col (str): Name of the target column.
    col_name (str): Name of the categorical column.
    ratio (bool, optional): If True, displays the ratio of unique values in the column. Default is True.

  Returns:
    None: Displays the combined figure with both plots.
  """
  fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 
  
  cat_distribution(df, col_name, axes[0], ratio)
  
  if col_name != target_col:
    target_by_cat(df, target_col, col_name, axes[1])
  else:
    axes[1].set_visible(False)
  
  plt.tight_layout()
  plt.show()
  
  return fig