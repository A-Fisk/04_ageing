# Figure 2 of the ageing project. Mean waveforms over the entire year
# with first and last month in the background?

# imports
import pathlib
import pandas as pd
idx = pd.IndexSlice
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.dates as mdates
import seaborn as sns
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.actogram_plot as aplot
import actiPy.periodogram as per
import actiPy.analysis as als

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

# shift the sj_post to line up with the lights of the rest
sj_fill_values = split_df_h.loc[idx["sj_post", :, "2010-01-01 00:00:00"], :]
# manual groupby to shift values up by one and fill in at the bottom
animals = split_df_h.index.get_level_values(1).unique()
group_list = []
for animal in animals:
    new_anim_df = split_df_h.loc[idx["sj_post", animal, :], :]
    anim_fill = sj_fill_values.loc[idx[:, animal, :], :]
    shifted_anim_df = new_anim_df.shift(-1)
    shifted_anim_df.iloc[-1] = anim_fill.values
    group_list.append(shifted_anim_df)
new_sj_post = pd.concat(group_list)
# set back in the main df
split_df_h.loc[idx["sj_post", :, :], :] = new_sj_post

# grab the mean values for the total time, as well as the first and last weeks
first_week = split_df_h.iloc[:, :8]
last_week = split_df_h.iloc[:, -7:]

total_mean = create_mean_df(split_df_h, mean_cols)
first_mean = create_mean_df(first_week, mean_cols)
last_mean = create_mean_df(last_week, mean_cols)

# Grab the light values for mean plots
light_df = split_df_h.loc[idx["dlan_post", "LDR", :], :].reset_index(
                                                         level=(0, 1),
                                                         drop=True)
light_mask = light_df > 300
lights = light_df.where(light_mask, other=500)
lights = lights.mask(light_mask, other=0)
lights = lights.mean(axis=1)

############ Calculate Qp IS IV

col_names = ["condition", "animal", "value"]
def longform_df(data, col_names):
    new_data = data.stack().reset_index()
    new_data.columns = col_names
    return new_data

# Calculate Qp
periodogram_power = all_df.groupby(level=[0]).apply(per.get_period,
                                                    return_power=True,
                                                    drop_lastcol=False)
# tidy output to get values
periodogram_power.columns = periodogram_power.columns.droplevel(1)
power_vals = periodogram_power.groupby(level=0).max()
power_vals.drop("LDR", axis=1, inplace=True)
power_vals = longform_df(power_vals, col_names)

# Calculate IS
# get number of days and divide power_vals by that
timespan = (df_list[0].index[-1] - df_list[0].index[0]).round("D").days
interdaily_stab = power_vals / timespan
interdaily_stab.drop("LDR", axis=1, inplace=True)
interdaily_stab = longform_df(interdaily_stab, col_names)

# Calculate IV
intraday_var = als.intradayvar(all_df, level=[0])
intraday_var.drop("LDR", axis=1, inplace=True)
intraday_var = longform_df(intraday_var, col_names)

# put together in a list
marker_dict = {
    "Qp": power_vals,
    "IS": interdaily_stab,
    "IV": intraday_var
}

###########
# plot

# plot constants
conditions = all_df.index.get_level_values(0).unique()
sections = ["total", "first_week", "last_week"]
alpha = 0.5
light_alpha = 0.1
mean_ylim = [0, 60]

# plot them
fig = plt.figure()

# create the subplots for the mean waveforms
wave_grid = gs.GridSpec(nrows=len(conditions), ncols=1, figure=fig, right=0.45)
wave_axes = [plt.subplot(x) for x in wave_grid]

# want to plot each condition as a separate row,
# with the first and last weeks all on the same axis
for condition, axis in zip(conditions, wave_axes):
       # select the data
    total_data = total_mean.loc[condition]
    first_data = first_mean.loc[condition]
    last_data = last_mean.loc[condition]

    # plot the data +/- sem for each bit
    for data, label in zip([total_data, first_data, last_data], sections):
        mean_data = data.iloc[:, 0]
        sem_data = data.iloc[:, 1]
        axis.plot(mean_data, label=label)
        axis.fill_between(data.index, (mean_data-sem_data),
                          (mean_data+sem_data), alpha=alpha)
        # plot the light as a background
        axis.fill_between(lights.index, 0, lights.values, alpha=light_alpha,
                          color='k')
        
    # fill in the background
    
    # tidy up titles and labels and limits
    if condition == conditions[0]:
        axis.legend()
    if condition != conditions[-1]:
        axis.set(xticklabels=[])
    mean_xlim = [total_data.index[0], total_data.index[-1]]
    axis.set(title=condition,
             ylim=mean_ylim,
             xlim=mean_xlim)

# marker values subplotting area
marker_grid = gs.GridSpec(nrows=3, ncols=1, figure=fig, left=0.55)
marker_axes = [plt.subplot(x) for x in marker_grid]

# make pretty
sns.set()

# plot each marker on a separate row
for marker_name, curr_axis in zip(marker_dict.keys(), marker_axes):
    
    curr_data = marker_dict[marker_name]


#  quick total per day
# sum = setup_df.groupby(level=0).sum().stack(level=0
#                 ).groupby(level=0).mean()


# TODO set x time format
# TODO add in total activity/variance measures in remaining subplots


plt.close('all')