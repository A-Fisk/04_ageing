# Figure 3. Episodes. Trying to decide whether population of episodes is
# different between the conditions

### Imports
import pathlib
import pandas as pd
idx = pd.IndexSlice
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns
sns.set()
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.analysis as als
import actiPy.periodogram as per
import actiPy.waveform as wave


### CONSTANTS
save_fig = pathlib.Path("/Users/angusfisk/Documents/"
                        "01_PhD_files/01_projects/01_thesisdata/04_ageing/"
                        "03_analysisoutputs/01_figures/03_fig3.png")
col_names = ["condition", "day", "animal", "measure"]
def longform(data, col_names):
    new_data = data.stack().reset_index()
    new_data.columns = col_names
    return new_data

### Step 1 Read in data
file_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects"
                        "/01_thesisdata/04_ageing"
                        "/01_datafiles/01_activity/02_episodes")

files = sorted(file_dir.glob("*.csv"))
file_names = [x.stem for x in files]
df_list = [prep.read_file_to_df(x, index_col=0) for x in files]

# hack to remove SJ PIR 1 and 2
sj_df = df_list[-1]
sj_df.iloc[:, 0:2] = np.nan
df_list[-1] = sj_df

# gather into a df
df_dict = dict(zip(file_names, df_list))
all_df = pd.concat(df_dict, sort=False)
all_df.drop("LDR", axis=1, inplace=True)

exp_data = all_df.loc[idx[:, "2017-10-12":"2017-11-23"], :]

### Step 2 calculate length per day and count per day
median_data = exp_data.groupby(level=0).resample("D", level=1).median()
bool_data = (exp_data > 0).astype(bool)
count_data = bool_data.groupby(level=0).resample("D", level=1).sum()
count_data.replace(0, np.nan, inplace=True)

median_cols = col_names.copy()
median_cols[-1] = "median"
long_median = longform(median_data, median_cols)
count_cols = col_names.copy()
count_cols[-1] = "count"
long_count = longform(count_data, count_cols)

scatter_data = long_median.copy()
scatter_data[count_cols[-1]] = long_count.iloc[:, -1]

minutes = exp_data / 60
hist_data = longform(minutes, col_names=col_names)

### Step 3 plot all

# plotting constants
conditions = exp_data.index.get_level_values(0).unique()
condition_col = col_names[0]
day_col = col_names[1]
animal_col = col_names[2]
median_col = median_cols[-1]
count_col = count_cols[-1]

# histogram labels
hist_xlabel = 'Episode duration (Minutes)'
hist_ylabel = "Normalised Density"
hist_title = "Distribution of Episode Duration"

# kde constants
kde_xlim = [0, 800]
kde_ylim = [20, 70]
kde_title = "Median length vs count of episodes per day"
kde_xlabel = "Median episode length per day, (S)"
kde_ylabel = "No. episodes / day"

title = "Distribution of activity episodes after long term disruption"

# Initialise the figure
fig = plt.figure()

###### Histogram plot
# create histogram axis - singular
hist_grid = gs.GridSpec(nrows=1, ncols=1, figure=fig, right=0.5)
hist_axes_list = [plt.subplot(x) for x in hist_grid]

# tidy data into list of arrays
hist_list = []
for condition in conditions:
    condition_data = hist_data[hist_data[condition_col] == condition]
    hist_list.append(condition_data[col_names[-1]])

# plot on axis
hist_axis = hist_axes_list[0]
hist_axis.hist(hist_list, density=True, label=conditions)
hist_axis.set_yscale('log')

# tidy axis
hist_axis.set(xlabel=hist_xlabel,
              ylabel=hist_ylabel,
              title=hist_title)
hist_axis.legend()

####### KDE plot
# create axis
kde_grid = gs.GridSpec(nrows=3, ncols=1, figure=fig, left=0.55,
                       hspace=0)
kde_axes_list = [plt.subplot(x) for x in kde_grid]

# grab the data for stats
median_data_list = []
count_data_list = []
# plot each condition
for condition, kde_axis in zip(conditions, kde_axes_list):
    kde_data = scatter_data[scatter_data[condition_col] == condition]
    median_data = kde_data[median_col]
    median_data_list.append(median_data)
    count_data = kde_data[count_col]
    count_data_list.append(count_data)
    sns.kdeplot(median_data, count_data, shade=True,
                shade_lowest=False, ax=kde_axis, label=condition)
    
    # label with condition
    kde_axis.text(1, 0.5, condition, rotation=270, color='k',
                  transform=kde_axis.transAxes)
    # tidy up axis
    kde_axis.set(xlim=kde_ylim,
                 ylim=kde_xlim,
                 ylabel=kde_ylabel)
    if condition != conditions[-1]:
        kde_axis.set(xlabel=[],
                     xticklabels="")
    if condition == conditions[0]:
        kde_axis.set(title=kde_title)
kde_axis.set(xlabel=kde_xlabel)

fig.suptitle(title)

fig.set_size_inches(11.69, 8.27)

plt.savefig(save_fig, dpi=600)

plt.close('all')