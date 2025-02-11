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
    None: The figure is saved to the specified folder.
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
    print(f"\n\n📌 Ratio of unique values in '{col_name}':")
    print(pd.DataFrame(df[col_name].value_counts(normalize=True) * 100).rename(columns={"proportion": "Ratio (%)"}).sort_values("Ratio (%)", ascending=False))
    print("-" * 50)
  
  sns.countplot(data=df, x=col_name, hue=col_name, palette="Set3", legend=False, order=df[col_name].value_counts().index, ax=ax)
  ax.set_title(f"Distribution of '{col_name}'")
  ax.set_xlabel(f"'{col_name}'")
  ax.set_ylabel("Count")
  
  if df[col_name].nunique() > 5:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
  
  
def num_distribution(df: pd.DataFrame, col_name: str, ax: plt.Axes, hue: str = None) -> None:
  """
  Visualizes the distribution of a numerical column in a given DataFrame.

  Args:
    df (pandas.DataFrame): DataFrame containing the column to visualize.
    col_name (str): Name of the column to visualize.
    ax (matplotlib.axes.Axes): Axes object to draw the plot.
    hue (str, optional): Name of the column to color the plot by. Default is None.

  Returns:
    None: Plot is drawn directly on the provided axes object.
  """
  if col_name not in df.columns:
    raise ValueError(f"Column '{col_name}' not found in DataFrame.")
  
  sns.histplot(data=df, x=col_name, kde=True, bins=30, ax=ax, hue=hue, palette="Set2")
  ax.set_title(f"Distribution of '{col_name}'{' by ' + hue if hue else ''}")
  ax.set_xlabel(f"'{col_name}'")
  

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


def target_by_num(df: pd.DataFrame, target_col: str, col_name: str, ax: plt.Axes, kind="bar") -> None:
  """
  Visualizes the target distribution by a numerical column in a given DataFrame.

  Args:
    df (pandas.DataFrame): DataFrame containing the columns to visualize.
    target_col (str): Name of the target column.
    col_name (str): Name of the numerical column.
    ax (matplotlib.axes.Axes): Axes object to draw the plot.
    kind (str, optional): Type of plot to draw ("violin" or "bar"). Default is "bar".

  Returns:
    None: Plot is drawn directly on the provided axes object.
  """
  if col_name not in df.columns:
    raise ValueError(f"Column '{col_name}' not found in DataFrame.")
  
  if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found in DataFrame.")
  
  if kind == "bar":
    sns.barplot(data=df, x=target_col, y=col_name, palette="coolwarm", ax=ax)
    ax.set_ylabel(f"Mean of '{col_name}'")
  elif kind == "violin":
    sns.violinplot(data=df, x=target_col, y=col_name, palette="Set1", ax=ax)
  else:
    raise ValueError("`kind` must be either 'bar' or 'violin'.")
  
  ax.set_title(f"Target ({target_col}) Distribution by {col_name}.")

  
def plot_cat_analysis(df: pd.DataFrame, col_name: str, target_col: str = None, ratio: bool = True) -> None:
  """
  Analyzes and visualizes a categorical feature of a given DataFrame.

  The function combines the following two plots into a single figure with two subplots:
  1. Distribution of the categorical feature.
  2. Target distribution by the categorical feature (if target_col is provided).

  Args:
    df (pandas.DataFrame): DataFrame containing the columns to visualize.
    col_name (str): Name of the categorical column.
    target_col (str, optional): Name of the target column. Default is None.
    ratio (bool, optional): If True, displays the ratio of unique values in the column. Default is True.

  Returns:
    None: Displays the combined figure with both plots.
  """
  fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 
  
  cat_distribution(df, col_name, axes[0], ratio)
  
  if target_col and col_name != target_col:
    target_by_cat(df, target_col, col_name, axes[1])
  else:
    axes[1].set_visible(False)
  
  plt.tight_layout()
  plt.show()
  
  return fig


def plot_num_analysis(df: pd.DataFrame, col_name: str, target_col: str = None, show_mean: bool = True) -> None:
  """
  Analyzes and visualizes a numerical feature of a given DataFrame.

  The function combines the following three plots into a single figure with three subplots:
  1. Distribution of the numerical feature.
  2. Numerical feature distribution by the categorical target feature (if target_col is provided).
  3. Violin plot of the numerical feature by the categorical target feature (if target_col is provided).

  Args:
    df (pandas.DataFrame): DataFrame containing the columns to visualize.
    col_name (str): Name of the numerical column.
    target_col (str, optional): Name of the categorical target column. Default is None.
    show_mean (bool, optional): If True, displays the mean of the numerical column by the target. Default is True.

  Returns:
    None: Displays the combined figure with both plots.
  """
  if show_mean and target_col:
    mean_by_target = df.groupby(target_col)[col_name].mean()
    print(f"\n\n📌 Mean of '{col_name}' by '{target_col}':\n{mean_by_target}")

  fig, axes = plt.subplots(1, 4, figsize=(20, 5))

  num_distribution(df, col_name, ax=axes[0])

  if target_col and col_name != target_col:
    num_distribution(df, col_name, hue=target_col, ax=axes[1])
    target_by_num(df, target_col, col_name, ax=axes[2])
    target_by_num(df, target_col, col_name, ax=axes[3], kind="violin")
  else:
    for i in range(1, 4):
      axes[i].set_visible(False)

  plt.tight_layout()
  plt.show()

  return fig
