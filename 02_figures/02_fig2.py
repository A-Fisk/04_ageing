# Figure 2 of the ageing project. Mean waveforms over the entire year
# with first and last month in the background?

# imports
import pathlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.dates as mdates
import seaborn as sns
sns.set()
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.actogram_plot as aplot

##################
# import the files
file_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects"
                        "/01_thesisdata/04_ageing"
                        "/01_datafiles/01_activity/01_post_disrupt")
save_dir = pathlib.Path("/Users/angusfisk/Documents/"
                        "01_PhD_files/01_projects/01_thesisdata/04_ageing/"
                        "03_analysisoutputs/01_figures")

files = sorted(file_dir.glob("*.csv"))
file_names = [x.stem for x in files]
df_list = [prep.read_file_to_df(x, index_col=0) for x in files]

######### Hack to set PIR 1 and 2 for sj to be Nan
sj_df = df_list[-1]
sj_df.iloc[:, 0:2] = np.nan
df_list[-1] = sj_df

df_dict = dict(zip(file_names, df_list))
all_df = pd.concat(df_dict, sort=False)

############################
# compute the mean waveforms

# split each animal and create dataframe out of it
split_dict = {}
for anim_no, anim_label in enumerate(all_df.columns):
    temp_split = all_df.groupby(level=0
                                ).apply(prep.split_dataframe_by_period,
                                        period="24H",
                                        animal_number=anim_no,
                                        reset_level=False)
    split_dict[anim_label] = temp_split
split_df = pd.concat(split_dict).reorder_levels([1, 0, 2]).sort_index()
split_df_h = split_df.groupby(level=[0, 1]).resample(
                                            "H", level=2).mean()

########## get the mean and sem values

def create_mean_df(data, cols):
    """takes in the split df and returns means and sems in a df"""
    # find the mean and sem
    setup_df = data.unstack(level=1
                    ).reorder_levels([1, 0], axis=1
                        ).sort_index(axis=1).drop("LDR", axis=1)
    means = setup_df.mean(axis=1)
    sems = setup_df.sem(axis=1)

    means_df = pd.concat([means, sems], axis=1)
    means_df.columns = cols
    
    return means_df

mean_cols = ["Means", "SEMS"]

# grab the mean values for the total time, as well as the first and last weeks
first_week = split_df_h.iloc[:, :8]
last_week = split_df_h.iloc[:, -7:]

total_mean = create_mean_df(split_df_h, mean_cols)
first_mean = create_mean_df(first_week, mean_cols)
last_mean = create_mean_df(last_week, mean_cols)

###########
# plot

# plot constants
conditions = all_df.index.get_level_values(0).unique()
alpha = 0.5

# plot them
fig = plt.figure()

# create the subplots for the mean waveforms
wave_grid = gs.GridSpec(nrows=len(conditions), ncols=1, figure=fig, right=0.5)
wave_axes = [plt.subplot(x) for x in wave_grid]

# want to plot each condition as a separate row,
# with the first and last weeks all on the same axis
for condition, axis in zip(conditions, wave_axes):
    
    # select the data
    total_data = total_mean.loc[condition]
    first_data = first_mean.loc[condition]
    last_data = last_mean.loc[condition]

    # plot the data +/- sem for each bit
    for data in [total_data, first_data, last_data]:
        mean_data = data.iloc[:, 0]
        sem_data = data.iloc[:, 1]
        axis.plot(mean_data)
        axis.fill_between(data.index, (mean_data-sem_data),
                          (mean_data+sem_data), alpha=alpha)
        
    # fill in the background
    
    # tidy up titles and labels and limits
    axis.legend()

total_mean.unstack(level=0).plot()
first_mean.unstack(level=0).plot()
last_mean.unstack(level=0).plot()



#  quick total per day
sum = setup_df.groupby(level=0).sum().stack(level=0
                ).groupby(level=0).mean()


# TODO add in lights for means
# TODO add in total activity/variance measures in remaining subplots


plt.close('all')