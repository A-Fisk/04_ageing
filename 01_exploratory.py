
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
import sleepPy.preprocessing as prep
import sleepPy.plots as plot
import actiPy.actogram_plot as aplot

pir_files_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/09_pirdata')

exp_filesdir = pir_files_dir / "03_experiment_files"


exp = sorted(exp_filesdir.glob("*5*"))

file = exp[2]
df = pd.read_csv(file,
                 index_col=[0],
                 parse_dates=True)

df.sort_index(inplace=True)
aplot._actogram_plot_from_df(df, 3, drop_level=False, fname=file)



plt.close()
