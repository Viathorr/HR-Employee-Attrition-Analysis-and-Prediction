import os
import matplotlib.pyplot as plt
import seaborn as sns

FIGS_DIR = os.path.abspath(os.path.join("..", "..", "reports", "figures"))

def save_figure(fig, filename, folder=FIGS_DIR):
  """
  Saves a given figure to a specified folder.

  Args:
    fig (matplotlib.figure.Figure): Figure object to save.
    filename (str): Name of the file (without extension).
    folder (str): Directory to save the file.

  Returns:
    None
  """
  os.makedirs(folder, exist_ok=True)
  save_path = os.path.join(folder, f"{filename}.png")
  
  fig.savefig(save_path, bbox_inches='tight')
  
  print(f"✅ Figure saved at `{save_path}`.")
