# Figure 2 of the ageing project. Mean waveforms over the entire year
# with first and last month in the background?

# imports
import pathlib
import pandas as pd
idx = pd.IndexSlice
import numpy as np
import pingouin as pg
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

save_fig = save_dir / "02_fig2.png"

files = sorted(file_dir.glob("*.csv"))
file_names = [x.stem for x in files]
df_list = [prep.read_file_to_df(x, index_col=0) for x in files]

######### Hack to set PIR 1 and 2 for sj to be Nan
sj_df = df_list[-1]
sj_df.iloc[:, 0:2] = np.nan
df_list[-1] = sj_df

df_dict = dict(zip(file_names, df_list))
all_df = pd.concat(df_dict, sort=False)

################################################################################
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

############ Calculate Qp IS IV ################################################

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

# Calculate IS
# get number of days and divide power_vals by that
timespan = (df_list[0].index[-1] - df_list[0].index[0]).round("D").days
interdaily_stab = power_vals / timespan
long_power_vals = longform_df(power_vals, col_names)
long_interdaily_stab = longform_df(interdaily_stab, col_names)

# Calculate IV
intraday_var = als.intradayvar(all_df, level=[0])
intraday_var.drop("LDR", axis=1, inplace=True)
long_intraday_var = longform_df(intraday_var, col_names)

# put together in a list
marker_dict = {
    "Qp": long_power_vals,
    "IS": long_interdaily_stab,
    "IV": long_intraday_var
}

# Values for total activity per day ############################################

# get the values
daily = all_df.groupby(level=0).resample("D", level=1).mean()
daily.drop("LDR", axis=1, inplace=True)
daily_mean = daily.mean(axis=1).unstack(level=0)
daily_sem = daily.sem(axis=1).unstack(level=0)


################################################################################
# stats

stats_colnames = ["Protocol", "Hour", "Animal", "Value"]
protocol_col = stats_colnames[0]
anim_col = stats_colnames[2]
hour_col = stats_colnames[1]
dep_var = stats_colnames[3]
hours = split_df_h.index.get_level_values(-1).unique()

save_test_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                             "01_projects/01_thesisdata/04_ageing/"
                             "03_analysisoutputs/01_figures/00_csvs/02_fig2")
anova_str = "01_anova.csv"
ph_str = "02_posthoc.csv"

# Q1. Is the mean activity profile different between groups?
# 2 way anova activity ~ protocol*hour with ph activity ~ protocol | hour

mean_test_dir = save_test_dir / "01_mean"
split_labels = ["Total", "First", "Last"]
for split_df, label in zip([split_df_h, first_week, last_week], split_labels):
    print(label)
    label_test_dir = mean_test_dir / label
    # tidy data
    test_df = split_df.mean(axis=1)
    test_df.drop("LDR", level=1, inplace=True)
    label_df = test_df.unstack(level=1
                               ).groupby(level=0
                                         ).apply(prep.label_anim_cols)
    tidy_df = label_df.stack().reset_index()
    tidy_df.columns = stats_colnames
    print(tidy_df.head())

    # perform anova
    anova = pg.mixed_anova(dv=dep_var,
                           between=protocol_col,
                           within=hour_col,
                           subject=anim_col,
                           data=tidy_df)
    pg.print_table(anova)

    # do post hocs
    ph_df = prep.tukey_pairwise_ph(tidy_df)
    
    anova.to_csv((label_test_dir / anova_str))
    ph_df.to_csv((label_test_dir / ph_str))
    

# Q2. Does the condition affect Qp/IV/IS?
# 1 way anova of values for each

marker_test_dir = save_test_dir / "02_markers"
for label, df in zip(marker_dict.keys(),
                     [power_vals, interdaily_stab, intraday_var]):
    print(label)
    marker_label_test_dir = marker_test_dir / label
    marker_df = df
    label_df = marker_df.groupby(level=0
                                 ).apply(prep.label_anim_cols
                                         ).stack(
                                                 ).reset_index()
    label_df.columns = [stats_colnames[x] for x in [0, 2, 3]]
    print(label_df.head())
    
    # anova
    marker_anova = pg.anova(dv=dep_var,
                            between=protocol_col,
                            data=label_df)
    pg.print_table(marker_anova)
    
    marker_anova.to_csv((marker_label_test_dir / anova_str))

# Q3. Does the total activity change between groups?
# 2 way anova of activity ~ day*protocol with activity ~ protocol | day

tot_act_test_dir = save_test_dir / "03_tot_act"
act_df = daily
label_df = act_df.groupby(level=0
                          ).apply(prep.label_anim_cols
                                  ).stack(
                                          ).reset_index()
label_df.columns = stats_colnames
print(label_df.head())

# anova
activity_anova = pg.mixed_anova(dv=dep_var,
                                between=protocol_col,
                                within=hour_col,
                                subject=anim_col,
                                data=label_df)
pg.print_table(activity_anova)

ph = prep.tukey_pairwise_ph(label_df)

activity_anova.to_csv((tot_act_test_dir / anova_str))
ph.to_csv((tot_act_test_dir / ph_str))


################################################################################
# plot

# plot constants
conditions = all_df.index.get_level_values(0).unique()
sections = ["total", "first_week", "last_week"]
alpha = 0.5
light_alpha = 0.1
mean_ylim = [0, 60]
condition_col = col_names[0]
animal_col = col_names[1]
value_col = col_names[2]
xfmt = mdates.DateFormatter("%H:%M:%S")
interval = 3
left_xlabel = "Circadian time (?) (Hrs)"
fig_title = "Differences between groups when returned to LD"
capsize = 5
activity_title = "Total activity per day"
mean_title = "mean activity over 24 hours"

# plot them
fig = plt.figure()

# create the subplots for the mean waveforms
wave_grid = gs.GridSpec(nrows=len(conditions), ncols=1, figure=fig, right=0.45)
wave_axes = [plt.subplot(x) for x in wave_grid]

# want to plot each condition as a separate row,
# with the first and last weeks all on the same axis
data_dict = {
    "total": total_mean,
    "first": first_mean,
    "last": last_mean
}
for data_label, axis in zip(data_dict.keys(), wave_axes):
    
    data = data_dict[data_label]
    
    # plot the data +/- sem for each bit
    for condition in conditions:
        # select the data
        curr_data = data.loc[condition]
        mean_data = curr_data.iloc[:, 0]
        sem_data = curr_data.iloc[:, 1]
        axis.plot(mean_data, label=condition)
        axis.fill_between(curr_data.index, (mean_data-sem_data),
                          (mean_data+sem_data), alpha=alpha)
        # plot the light as a background
        axis.fill_between(lights.index, 0, lights.values, alpha=light_alpha,
                          color='k')
        
    # fill in the background
    
    # tidy up titles and labels and limits
    if data_label == list(data_dict.keys())[0]:
        axis.legend()
        axis.set(title=mean_title)
    if data_label != list(data_dict.keys())[-1]:
        axis.set(xticklabels=[])
    mean_xlim = [(curr_data.index[0] - pd.Timedelta("10M")), (curr_data.index[
        -1])]
    axis.set(ylabel=data_label,
             ylim=mean_ylim,
             xlim=mean_xlim)
    axis.xaxis.set_major_formatter(xfmt)
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
axis.set(xlabel=left_xlabel)

# plot the change in total activity over time
activity_grid = gs.GridSpec(nrows=1, ncols=1, figure=fig, left=0.55, top=0.45)
activity_axes = [plt.subplot(x) for x in activity_grid]
ac_axis = activity_axes[0]

# plot on the axis
for condition in daily_mean.columns:
    mean = daily_mean.loc[:, condition]
    sem = daily_sem.loc[:, condition]
    ac_axis.errorbar(daily_mean.index, mean, yerr=sem, capsize=capsize,
                     label=condition)
ac_axis.legend()
ac_axis.set(title=activity_title)

fig.autofmt_xdate()

# marker values subplotting area
marker_grid = gs.GridSpec(nrows=3, ncols=1, figure=fig, left=0.55, hspace=0,
                          bottom=0.55)
marker_axes = [plt.subplot(x) for x in marker_grid]

# plot each marker on a separate row
for marker_name, curr_axis in zip(marker_dict.keys(), marker_axes):
    
    curr_data = marker_dict[marker_name]
    sns.boxplot(x=condition_col, y=value_col, data=curr_data,
                ax=curr_axis, fliersize=0)
    sns.stripplot(x=condition_col, y=value_col, data=curr_data,
                  ax=curr_axis, dodge=True, color='k')
    
    # add in labels and tidy plot
    curr_axis.set(ylabel=marker_name)
    
    # remove x values if not the bottom
    if marker_name != list(marker_dict.keys())[-1]:
        curr_axis.set(xlabel=[],
                      xticklabels=[])
curr_axis.set(xlabel="")

fig.suptitle(fig_title)

fig.set_size_inches(11.69, 8.27)

plt.savefig(save_fig, dpi=600)

plt.close('all')
