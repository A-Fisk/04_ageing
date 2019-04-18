#, Figure 2 of the ageing project. Mean waveforms over the entire year
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
sys.path.insert(
    0,
    "/Users/angusfisk/Documents/01_PhD_files/"
    "07_python_package/actiPy"
)
import actiPy.preprocessing as prep
import actiPy.actogram_plot as aplot
import actiPy.periodogram as per
import actiPy.analysis as als

##################
# import the files
file_dir = pathlib.Path(
    "/Users/angusfisk/Documents/01_PhD_files/01_projects"
    "/01_thesisdata/04_ageing"
    "/01_datafiles/01_activity/01_post_disrupt"
)
save_dir = pathlib.Path(
    "/Users/angusfisk/Documents/"
    "01_PhD_files/01_projects/01_thesisdata/04_ageing/"
    "03_analysisoutputs/01_figures"
)

save_fig = save_dir / "02_fig2.png"

files = sorted(file_dir.glob("*.csv"))
file_names = [x.stem[:-5].upper() for x in files]
df_list = [prep.read_file_to_df(x, index_col=0) for x in files]

######### Hack to set PIR 1 and 2 for sj to be Nan
sj_df = df_list[-1]
sj_df.iloc[:, 0:2] = np.nan
df_list[-1] = sj_df

df_dict = dict(zip(file_names, df_list))
all_df = pd.concat(df_dict, sort=False)

# constants
conditions = all_df.index.get_level_values(0).unique()

# remove the first 14 days so all re-entrained
time_remove_start = pd.Timedelta("14D")
all_df = all_df.loc[
    idx[:, (all_df.loc["DLAN"].index[0] + time_remove_start):],
    :
]

# find out how much sj is delayed by
all_lights_on = all_df[all_df.loc[:, "LDR"] > 100]
dlan_lights_on = all_lights_on.loc["DLAN"].first_valid_index()
sj_lights_on = all_lights_on.loc["SJ"].first_valid_index()
shift_amount = len(
    all_df.loc[
        idx["SJ", dlan_lights_on.round("T"):sj_lights_on.round("T")],
        :
    ]
)
sj_shifted = all_df.loc["SJ"].shift(-shift_amount)
# replace all df with shifted values
all_df.loc["SJ"].update(sj_shifted)

# shift start to the first lights on
all_df = all_df.loc[
    idx[:, dlan_lights_on.round("T"):"2019"],
    :
]


############ Calculate Markers  ################################################

col_names = ["Condition", "Animal", "Value"]
calc_df = all_df.loc[:, :"PIR6"]
def longform_df(data, col_names):
    new_data = data.stack().reset_index()
    new_data.columns = col_names
    return new_data


# Calculate LS Periodogram power
periodogram_power = calc_df.groupby(
    level=[0]
).apply(
    per.get_period,
    return_power=True,
    return_periods=False,
    drop_lastcol=False
)
# tidy output to get values
periodogram_power.columns = periodogram_power.columns.droplevel(1)
power_vals = periodogram_power.groupby(level=0).max()
power_long = longform_df(power_vals, col_names)


# Calculate IS
# get number of days and divide power_vals by that
timespan = (
    calc_df.loc[conditions[0]].index[-1] -
    calc_df.loc[conditions[0]].index[0]
).round("D").days
interdaily_stab = power_vals / timespan
is_long = longform_df(interdaily_stab, col_names)


# Calculate IV
intraday_var = als.intradayvar(calc_df, level=[0])
iv_long = longform_df(intraday_var, col_names)


# Calculate Lightphase Activity
light_act = all_df.groupby(
    level=0
).apply(
    als.light_phase_activity_nfreerun
)
light_act_long = longform_df(light_act.iloc[:, :-1], col_names)

# Calculate Relative Amplitude on hourly daily mean
def split_in_groupby(test_df, **kwargs):
    curr_split_dict = {}
    for animal_no, animal_label in enumerate(test_df.columns):
        curr_split = prep.split_dataframe_by_period(
            test_df,
            animal_number=animal_no,
            drop_level=True,
            reset_level=False
        )
        curr_split_dict[animal_label] = curr_split
    curr_split_df = pd.concat(curr_split_dict)
    
    return curr_split_df

split_df = all_df.groupby(
    level=0
).apply(
    split_in_groupby
)
split_df_hourly = split_df.groupby(
    level=[0, 1]
).resample(
    "H",
    level=2,
    loffset=pd.Timedelta("30M")
).mean()
split_daily_mean = split_df_hourly.mean(axis=1)
split_daily_tidy = split_daily_mean.unstack(level=1)
split_daily_calc = split_daily_tidy.drop("LDR", axis=1)

rel_amp = split_daily_calc.groupby(
    level=0
).apply(
    als.relative_amplitude
)
rel_amp_long = longform_df(rel_amp, col_names)

# Calculate total activity per day
tot_act_days = calc_df.groupby(
    level=0
).resample(
    "D",
    level=1
).sum()
tot_act_days[tot_act_days == 0] = np.nan
tot_act = tot_act_days.groupby(
    level=0
).mean()
# tot_act[tot_act == 0 ] = np.nan
tot_act_long = longform_df(tot_act, col_names)

# Bring together in a collection
marker_dict = {
    "Intradaily Variability": iv_long,
    "Power": power_long,
    "Interdaily Stability": is_long,
    "Lightphase activity": light_act_long,
    "Relative amplitude": rel_amp_long,
    "Total activity": tot_act_long
}

# Values for total activity per day ############################################

# get the values
daily = all_df.groupby(level=0).resample("D", level=1).mean()
daily.drop("LDR", axis=1, inplace=True)
daily_mean = daily.mean(axis=1).unstack(level=0)
daily_sem = daily.sem(axis=1).unstack(level=0)
################################################################################
# # compute the mean waveforms
#
# # split each animal and create dataframe out of it
# split_dict = {}
# for anim_no, anim_label in enumerate(all_df.columns):
#     temp_split = all_df.groupby(level=0
#                                 ).apply(prep.split_dataframe_by_period,
#                                         period="24H",
#                                         animal_number=anim_no,
#                                         reset_level=False)
#     split_dict[anim_label] = temp_split
# split_df = pd.concat(split_dict).reorder_levels([1, 0, 2]).sort_index()
# split_df_h = split_df.groupby(level=[0, 1]).resample(
#                                             "H", level=2).mean()
#
# ########## get the mean and sem values
#
# def create_mean_df(data, cols):
#     """takes in the split df and returns means and sems in a df"""
#     # find the mean and sem
#     setup_df = data.unstack(level=1
#                     ).reorder_levels([1, 0], axis=1
#                         ).sort_index(axis=1).drop("LDR", axis=1)
#     means = setup_df.mean(axis=1)
#     sems = setup_df.sem(axis=1)
#
#     means_df = pd.concat([means, sems], axis=1)
#     means_df.columns = cols
#
#     return means_df
#
# mean_cols = ["Means", "SEMS"]
#
# # shift the sj_post to line up with the lights of the rest
# sj_fill_values = split_df_h.loc[idx["sj_post", :, "2010-01-01 00:00:00"], :]
# # manual groupby to shift values up by one and fill in at the bottom
# animals = split_df_h.index.get_level_values(1).unique()
# group_list = []
# for animal in animals:
#     new_anim_df = split_df_h.loc[idx["sj_post", animal, :], :]
#     anim_fill = sj_fill_values.loc[idx[:, animal, :], :]
#     shifted_anim_df = new_anim_df.shift(-1)
#     shifted_anim_df.iloc[-1] = anim_fill.values
#     group_list.append(shifted_anim_df)
# new_sj_post = pd.concat(group_list)
# # set back in the main df
# split_df_h.loc[idx["sj_post", :, :], :] = new_sj_post
#
# # grab the mean values for the total time, as well as the first and last weeks
# first_week = split_df_h.iloc[:, :8]
# last_week = split_df_h.iloc[:, -7:]
#
# total_mean = create_mean_df(split_df_h, mean_cols)
# first_mean = create_mean_df(first_week, mean_cols)
# last_mean = create_mean_df(last_week, mean_cols)
#
# # Grab the light values for mean plots
# light_df = split_df_h.loc[idx["dlan_post", "LDR", :], :].reset_index(
#                                                          level=(0, 1),
#                                                          drop=True)
# light_mask = light_df > 300
# lights = light_df.where(light_mask, other=500)
# lights = lights.mask(light_mask, other=0)
# lights = lights.mean(axis=1)
#
#


################################################################################
# # stats
#
# stats_colnames = ["Protocol", "Hour", "Animal", "Value"]
# protocol_col = stats_colnames[0]
# anim_col = stats_colnames[2]
# hour_col = stats_colnames[1]
# dep_var = stats_colnames[3]
# hours = split_df_h.index.get_level_values(-1).unique()
#
# save_test_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
#                              "01_projects/01_thesisdata/04_ageing/"
#                              "03_analysisoutputs/01_figures/00_csvs/02_fig2")
# anova_str = "01_anova.csv"
# ph_str = "02_posthoc.csv"
#
# # Q1. Is the mean activity profile different between groups?
# # 2 way anova activity ~ protocol*hour with ph activity ~ protocol | hour
#
# mean_test_dir = save_test_dir / "01_mean"
# split_labels = ["Total", "First", "Last"]
# for split_df, label in zip([split_df_h, first_week, last_week], split_labels):
#     print(label)
#     label_test_dir = mean_test_dir / label
#     # tidy data
#     test_df = split_df.mean(axis=1)
#     test_df.drop("LDR", level=1, inplace=True)
#     label_df = test_df.unstack(level=1
#                                ).groupby(level=0
#                                          ).apply(prep.label_anim_cols)
#     tidy_df = label_df.stack().reset_index()
#     tidy_df.columns = stats_colnames
#     print(tidy_df.head())
#
#     # perform anova
#     anova = pg.mixed_anova(dv=dep_var,
#                            between=protocol_col,
#                            within=hour_col,
#                            subject=anim_col,
#                            data=tidy_df)
#     pg.print_table(anova)
#
#     # do post hocs
#     ph_df = prep.tukey_pairwise_ph(tidy_df)
#
#     anova.to_csv((label_test_dir / anova_str))
#     ph_df.to_csv((label_test_dir / ph_str))
#
#
# # Q2. Does the condition affect Qp/IV/IS?
# # 1 way anova of values for each
#
# marker_test_dir = save_test_dir / "02_markers"
# for label, df in zip(marker_dict.keys(),
#                      [power_vals, interdaily_stab, intraday_var]):
#     print(label)
#     marker_label_test_dir = marker_test_dir / label
#     marker_df = df
#     label_df = marker_df.groupby(level=0
#                                  ).apply(prep.label_anim_cols
#                                          ).stack(
#                                                  ).reset_index()
#     label_df.columns = [stats_colnames[x] for x in [0, 2, 3]]
#     print(label_df.head())
#
#     # anova
#     marker_anova = pg.anova(dv=dep_var,
#                             between=protocol_col,
#                             data=label_df)
#     pg.print_table(marker_anova)
#
#     marker_anova.to_csv((marker_label_test_dir / anova_str))
#
# # Q3. Does the total activity change between groups?
# # 2 way anova of activity ~ day*protocol with activity ~ protocol | day
#
# tot_act_test_dir = save_test_dir / "03_tot_act"
# act_df = daily
# label_df = act_df.groupby(level=0
#                           ).apply(prep.label_anim_cols
#                                   ).stack(
#                                           ).reset_index()
# label_df.columns = stats_colnames
# print(label_df.head())
#
# # anova
# activity_anova = pg.mixed_anova(dv=dep_var,
#                                 between=protocol_col,
#                                 within=hour_col,
#                                 subject=anim_col,
#                                 data=label_df)
# pg.print_table(activity_anova)
#
# ph = prep.tukey_pairwise_ph(label_df)
#
# activity_anova.to_csv((tot_act_test_dir / anova_str))
# ph.to_csv((tot_act_test_dir / ph_str))


################################################################################
# plot

# Plot all on the same figure
fig = plt.figure()

# plot markers
marker_grid = gs.GridSpec(
    nrows=len(marker_dict),
    ncols=1,
    figure=fig,
    right=0.5,
    # hspace=0
)
marker_axes = [plt.subplot(x) for x in marker_grid]

# Marker constants
dep_var = col_names[-1]
condition_col = col_names[0]
anim_col = col_names[1]
condition_order = [conditions[1], conditions[0], conditions[2]]
# marker_colours = ["C0", "C1", "C2"]
# colour_dict_markers = dict(zip(condition_col, marker_colours))

# Tidy marker constants
capsize = 0.2
errwidth = 1
marker_size = 3
sem = 68

# marker label constants
ylabel_size = 10
marker_title_size = 12

for marker, curr_ax_marker in zip(marker_dict.keys(), marker_axes):
    
    curr_marker_df = marker_dict[marker]
    # curr_marker_colour = colour_dict_markers[marker]
    
    sns.pointplot(
        y=dep_var,
        x=condition_col,
        hue=condition_col,
        data=curr_marker_df,
        ax=curr_ax_marker,
        join=False,
        capsize=capsize,
        errwidth=errwidth,
        legend=False,
        order=condition_order,
        ci=sem
    )
    sns.swarmplot(
        y=dep_var,
        x=condition_col,
        data=curr_marker_df,
        ax=curr_ax_marker,
        size=marker_size,
        color='k',
        order=condition_order
    )
    curr_ax_marker.axvline(
        0.5,
        color='k'
    )
    curr_ax_marker.axvline(
        1.5,
        color='k'
    )
    
    # Tidy axes and labels
    curr_marker_legend = curr_ax_marker.legend()
    curr_marker_legend.remove()
    curr_ax_marker.set_ylabel(marker, fontsize=ylabel_size)
    curr_ax_marker.ticklabel_format(
        axis='y',
        scilimits=(-2, 2)
    )
    if marker != list(marker_dict.keys())[-1]:
        curr_ax_marker.set_xticklabels("")
        curr_ax_marker.set_xlabel("")
    
fig.align_ylabels(marker_axes)
marker_axes[0].text(
    0.5,
    1.1,
    "Circadian Markers",
    ha='center',
    fontsize=marker_title_size,
    transform=marker_axes[0].transAxes
)


# Plot mean activity over the day
wave_grid = gs.GridSpec(
    nrows=1,
    ncols=1,
    figure=fig,
    left=0.55,
    bottom=0.55
)
wave_axes = [plt.subplot(x) for x in wave_grid]

curr_axis_wave = wave_axes[0]

# Tidy wave constants
wave_alpha = 0.5
dark_index = pd.DatetimeIndex(
    start="2010-01-01 12:00:00",
    end='2010-01-02 02:00:00',
    freq="H"
)
start_index = pd.Timestamp("2010-01-01 00:00:00")
end_index = pd.Timestamp("2010-01-02 00:00:00")
xfmt = mdates.DateFormatter("%H:%M")
fontsize_time = 8

for condition in conditions:
    curr_data_wave = split_daily_calc.loc[condition]
    curr_data_mean = curr_data_wave.mean(axis=1)
    curr_data_sem = curr_data_wave.sem(axis=1)
    
    
    curr_axis_wave.plot(
        curr_data_wave.index,
        curr_data_mean,
        label=condition
    )
    curr_axis_wave.fill_between(
        curr_data_wave.index,
        curr_data_mean - curr_data_sem,
        curr_data_mean + curr_data_sem,
        alpha=wave_alpha
    )
    
curr_axis_wave.fill_between(
    dark_index,
    500,
    alpha=lights_alpha
)

# Tidy axes and labels
curr_axis_wave.xaxis.set_major_formatter(xfmt)
curr_axis_wave.set_ylim([0, 60])
curr_axis_wave.set_xlim(
    start_index,
    end_index
)
for label in curr_axis_wave.get_xticklabels():
    label.set_ha('right')
    label.set_rotation(30)
    label.set_fontsize(fontsize_time)
curr_axis_wave.set_xlabel(
    "Time, ZT, Hours"
)
curr_axis_wave.set_ylabel(
    "Mean activity"
)
curr_axis_wave.set_title(
    "Mean daily activity"
)


######## Plot total activity per day
total_grid = gs.GridSpec(
    nrows=1,
    ncols=1,
    figure=fig,
    left=0.55,
    top=0.45
)
total_axes = [plt.subplot(x) for x in total_grid]

curr_axis_total = total_axes[0]

# Tidy total plot constants
bins_per_day = 8640
capsize_totals = 5

for condition in conditions:
    curr_data_total = tot_act_days.loc[condition]
    curr_data_total = curr_data_total.iloc[1:].copy()
    curr_tot_mean = curr_data_total.mean(axis=1)
    curr_tot_sem = curr_data_total.sem(axis=1)
    
    curr_axis_total.errorbar(
        curr_data_total.index,
        curr_tot_mean,
        yerr=curr_tot_sem,
        capsize=capsize_totals
    )

# Tidy axes
curr_axis_total.set_xlim(
    curr_data_total.index[0] - pd.Timedelta('1D'),
    curr_data_total.index[-1] + pd.Timedelta("1D")
)
total_xticks = curr_data_total.index[::4]
total_xticklabels = [(x - total_xticks[0]).days for x in total_xticks]
curr_axis_total.set_xticks(total_xticks)
curr_axis_total.set_xticklabels(total_xticklabels)
curr_axis_total.ticklabel_format(
    axis='y',
    scilimits=(0, 0)
)
curr_axis_total.set_xlabel(
    "Days"
)
curr_axis_total.set_ylabel(
    "Daily Activity, au"
)

# add in final touches to the figure
handles, labels = curr_axis_wave.get_legend_handles_labels()
fig.legend(
    handles=handles,
    loc=(0.9, 0.82),
    fontsize=10,
    markerscale=0.5
)

fig.suptitle(
    "Group Housed Circadian Parameters"
)

fig.set_size_inches(11.69, 8.27)

plt.savefig(save_fig, dpi=600)

plt.close('all')
