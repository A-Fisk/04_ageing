#, Figure 3/4 of the ageing project - singly housed circadian data

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
import os
import sys
sys.path.insert(
    0,
    "/Users/angusfisk/Documents/01_PhD_files/"
    "07_python_package/actiPy"
)
import actiPy.preprocessing as prep
import actiPy.plots as plot
import actiPy.periodogram as per
import actiPy.analysis as als

##################
# import the files
file_dir = pathlib.Path(
    "/Users/angusfisk/Documents/01_PhD_files/01_projects"
    "/01_thesisdata/04_ageing"
    "/01_datafiles/01_activity/03_single_housed/01_concat"
)
save_dir = pathlib.Path(
    "/Users/angusfisk/Documents/"
    "01_PhD_files/01_projects/01_thesisdata/04_ageing/"
    "03_analysisoutputs/01_figures"
)

save_fig = save_dir / "05_fig5.png"

files = sorted(file_dir.glob("*.csv"))
file_names = [x.stem.upper() for x in files]
df_list = [prep.read_file_to_df(x, index_col=[0, 1]) for x in files]
df_dict = dict(zip(file_names, df_list))
all_df = pd.concat(df_dict, sort=False)

# PIRS 6-11 look like they had LL for a few weeks from the actograms
# No notes to support what actually happened but unsure so excluding
all_df.loc[idx["LD"], "7":"12"] = np.nan
all_df.loc[idx["SJ"], ["10", "8"]] = np.nan # PIRs stopped working
all_df.loc[idx["DLAN"], "4"] = np.nan # LS not picking up correct period


# just use DD
light_df = all_df.loc[idx[:, "dd", :], :].copy()
light_df.index = light_df.index.droplevel(1)
all_df = light_df.copy()

# shift sj by an extra hour
dlan_lights_on = pd.Timestamp("2018-02-16 03:55:50")
sj_lights_on = pd.Timestamp("2018-02-16 04:55:30")
shift_amount = len(
    all_df.loc[
        idx["SJ", dlan_lights_on.round("T"):sj_lights_on.round("T")],
        :
    ]
)
sj_shifted = all_df.loc["SJ"].shift(-shift_amount)
# replace all df with shifted values
all_df.loc["SJ"].update(sj_shifted)

# shift to start of first cycle
all_df = all_df.loc[
    idx[:, dlan_lights_on.round("T"):"2019"],
    :
]

dlan_lights_on = pd.Timestamp("2017-12-22 03:55:50")
all_df = all_df.loc[
    idx[:, dlan_lights_on.round("T"):"2019"],
    :
]

# constants
conditions = all_df.index.get_level_values(0).unique()

############ Calculate Markers  ################################################

col_names = ["Condition", "Animal", "Value"]
calc_df = all_df.loc[:, :"12"]
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
periods = calc_df.groupby(
    level=[0]
).apply(
    per.get_period,
    return_power=False,
    return_periods=True,
    drop_lastcol=False
)
# tidy output to get values
periodogram_power.columns = periodogram_power.columns.droplevel(1)
power_vals = periodogram_power.groupby(level=0).max()
power_long = longform_df(power_vals, col_names)

# Calculate IS
# get number of days and divide power_vals by that
interdaily_stab_dict = {}
for condition in conditions:
    curr_timespan = len(calc_df.loc[condition])
    curr_power_vals = power_vals.loc[condition]
    curr_is = curr_power_vals / curr_timespan
    interdaily_stab_dict[condition] = curr_is
interdaily_stab = pd.concat(interdaily_stab_dict)
is_long = interdaily_stab.reset_index()
is_long.columns = col_names

# Calculate IV
intraday_var = als.intradayvar(calc_df, level=[0])
iv_long = longform_df(intraday_var, col_names)


# Calculate Lightphase Activity
# start by splitting by period
# needs to add levels to index to get to work in fx
calc_dict_1 = {"level_1": calc_df}
calc_df_1= pd.concat(calc_dict_1)
calc_df_list = [calc_df_1.loc["level_1"]]
period_dict_1 = {"level_1": periods}
period_df = pd.concat(period_dict_1)
period_df[pd.isnull(period_df)] = pd.Timedelta("24H")
periods_csv = save_dir / "00_csvs/05_fig5/01_ddperiods.csv"
period_df.to_csv(periods_csv)

activity_split = prep.split_list_with_periods(
    name_df=period_df,
    df_list=calc_df_list
)
light_act = activity_split.groupby(
    level=[0, 1, 2]
).apply(
    als.light_phase_activity_freerun
)
light_act_anim_mean = light_act.iloc[:, :-1].mean(axis=1)
light_act_anim_mean.index = light_act_anim_mean.index.droplevel(0)
light_act_long = light_act_anim_mean.reset_index()
light_act_long.columns = col_names

# Calculate Relative Amplitude on hourly daily mean
act_split_clean = activity_split.copy()
act_split_clean.index = act_split_clean.index.droplevel(0)
split_df_hourly = act_split_clean.groupby(
    level=[0, 1]
).resample(
    "H",
    level=2,
    loffset=pd.Timedelta("30M")
).mean()
split_daily_mean = split_df_hourly.mean(axis=1)
split_daily_tidy = split_daily_mean.unstack(level=1)
# split_daily_calc = split_daily_tidy.drop("LDR", axis=1)
split_relabel = split_daily_tidy.groupby(
    level=0
).apply(
    prep.label_anim_cols
).stack().reorder_levels([0, 2, 1])
split_colnames = [col_names[0], col_names[1], "Hour", col_names[2]]
split_relabel = split_relabel.reset_index()
split_relabel.columns = split_colnames

rel_amp = split_daily_tidy.groupby(
    level=0
).apply(
    als.relative_amplitude
)
rel_amp_long = longform_df(rel_amp, col_names)

# Calculate total activity per day
tot_act_days = act_split_clean.groupby(
    level=[0, 1]
).resample(
    "D",
    level=2
).sum()
tot_act_days[tot_act_days == 0] = np.nan
tot_act = tot_act_days.mean(axis=1)
tot_act.index = tot_act.index.droplevel(2)
tot_act_long = tot_act.reset_index()
tot_act_long.columns = col_names

# tidy for stats too
tot_relabel = tot_act_days.iloc[:, :-1].stack().unstack(level=1).groupby(
    level=0
).apply(
    prep.label_anim_cols
).stack()
tot_relabel.index = tot_relabel.index.droplevel(1)
tot_cols = [col_names[0], "Day", col_names[1], col_names[2]]
tot_relabel = tot_relabel.reset_index()
tot_relabel.columns = tot_cols

# Bring together in a collection
marker_dict = {
    "Intradaily Variability": iv_long,
    "Power": power_long,
    "Interdaily Stability": is_long,
    "Lightphase activity": light_act_long,
    "Relative amplitude": rel_amp_long,
    "Total activity": tot_act_long
}

################################################################################
# stats

condition_col = col_names[0]
anim_col = col_names[1]
dep_var = col_names[2]
hours = split_daily_tidy.index.get_level_values(-1).unique()
save_test_dir = pathlib.Path(
    "/Users/angusfisk/Documents/01_PhD_files/"
    "01_projects/01_thesisdata/04_ageing/"
    "03_analysisoutputs/01_figures/00_csvs/05_fig5"
)
anova_str = "01_anova.csv"
ph_str = "02_posthoc.csv"

# Q1. Does the condition affect the marker value?
# One way ANOVA Value ~ Protocol with PH value | protocol

marker_test_dir = save_test_dir / "01_markers"
if not os.path.exists(marker_test_dir):
    os.mkdir(marker_test_dir)
    
marker_ph_dict = {}
for marker_label, marker_df in zip(marker_dict.keys(), marker_dict.values()):
    
    print(marker_label)
    # run anova
    curr_anova_marker = pg.anova(
        dv=dep_var,
        between=condition_col,
        data=marker_df
    )
    pg.print_table(curr_anova_marker)
    
    curr_ph_marker = pg.pairwise_tukey(
        dv=dep_var,
        between=condition_col,
        data=marker_df
    )
    pg.print_table(curr_ph_marker)
    marker_ph_dict[marker_label] = curr_ph_marker
    
    # save the files
    label_test_dir = marker_test_dir / marker_label
    if not os.path.exists(label_test_dir):
        os.mkdir(label_test_dir)
    curr_anova_marker.to_csv(label_test_dir / anova_str)
    curr_ph_marker.to_csv(label_test_dir / ph_str)

marker_ph_df = pd.concat(marker_ph_dict)
marker_ph_df.set_index(["A", "B"], append=True, inplace=True)

# Q2 Does the condition affect the mean activity profile?
# Two way mixed anova of activity ~ condition*hour
# followed by post-hoc test of activity ~ condition | hour
hour_col = split_relabel.columns[2]
mean_test_dir = save_test_dir / "02_mean_wave"
if not os.path.exists(mean_test_dir):
    os.mkdir(mean_test_dir)

mean_anova = pg.mixed_anova(
    dv=dep_var,
    between=condition_col,
    within=hour_col,
    subject=anim_col,
    data=split_relabel
)
pg.print_table(mean_anova)
mean_anova_str = mean_test_dir / anova_str
mean_anova.to_csv(mean_anova_str)

mean_posthoc = prep.tukey_pairwise_ph(
    split_relabel,
    protocol_col=condition_col
)
mean_ph_str = mean_test_dir / ph_str
mean_posthoc.to_csv(mean_ph_str)


# Q3 Does the condition affect the total activity per day
# Two way mixed anova of activity ~ condition*day
# Posthoc test of activity ~ conditin | day

tot_day_col = tot_cols[1]
tot_test_dir = save_test_dir / "03_total_activity"
if not os.path.exists(tot_test_dir):
    os.mkdir(tot_test_dir)


tot_anova = pg.mixed_anova(
    dv=dep_var,
    between=condition_col,
    within=tot_day_col,
    subject=anim_col,
    data=tot_relabel
)
pg.print_table(tot_anova)
tot_anova_str = tot_test_dir / anova_str
tot_anova.to_csv(tot_anova_str)

tot_posthoc = prep.tukey_pairwise_ph(
    tot_relabel,
    protocol_col=condition_col,
    hour_col=tot_day_col
)
tot_ph_str = tot_test_dir / ph_str
tot_posthoc.to_csv(tot_ph_str)

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

# stats constants
yleveldlan = 0.9
ylevelsj = 0.95

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
        hue_order=condition_order,
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
        scilimits=(-2, 3)
    )
    if marker != list(marker_dict.keys())[-1]:
        curr_ax_marker.set_xticklabels("")
        curr_ax_marker.set_xlabel("")
    
    # Add in stats - if significant at any level draw a line?
    
    # add in stats
    ycoordlan = plot.sig_line_coord_get(
        curr_ax_marker,
        yleveldlan
    )
    ycoordsj = plot.sig_line_coord_get(
        curr_ax_marker,
        ylevelsj
    )
    
    # get x value from tests
    curr_ph_marker = marker_ph_df.loc[idx[marker, :], :]
    xcoorddlan = plot.sig_locs_get(
        curr_ph_marker,
        index_level2val=0,
        index_levelget=2
    )
    xcoordsj = plot.sig_locs_get(
        curr_ph_marker,
        index_level2val=2,
        index_levelget=3
    )
    
    # get xdict to lookup
    label_loc_dict = plot.get_xtick_dict(curr_ax_marker)
    
    # plot on the axis
    plot.draw_sighlines(
        yval=ycoordlan,
        sig_list=xcoorddlan,
        label_loc_dict=label_loc_dict,
        minus_val=1,
        plus_val=0,
        color="C1",
        curr_ax=curr_ax_marker
    )
    plot.draw_sighlines(
        yval=ycoordsj,
        sig_list=xcoordsj,
        label_loc_dict=label_loc_dict,
        minus_val=2,
        plus_val=0,
        color="C2",
        curr_ax=curr_ax_marker
    )
    
    
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
lights_alpha = 0.5
dark_index = pd.DatetimeIndex(
    start="2010-01-01 12:00:00",
    end='2010-01-02 02:00:00',
    freq="H"
)
start_index = pd.Timestamp("2010-01-01 00:00:00")
end_index = pd.Timestamp("2010-01-02 00:00:00")
xfmt = mdates.DateFormatter("%H:%M")
fontsize_time = 8

for condition in condition_order:
    curr_data_wave = split_daily_tidy.loc[condition]
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
  
# add in stats
# ycoords
ycoord_dlan_wave = plot.sig_line_coord_get(curr_axis_wave, yleveldlan )
ycoord_sj_wave = plot.sig_line_coord_get(curr_axis_wave, ylevelsj)

# xcoords from marker
sig_vals_dlan = plot.sig_locs_get(
    mean_posthoc,
    index_level2val=0,
    index_levelget=0
)
sig_vals_sj = plot.sig_locs_get(
    mean_posthoc,
    index_level2val=2,
    index_levelget=0
)

# plot xvals
minus_val = pd.Timedelta("15M")
xvals_dlan = [plot.get_xval_dates(
    x,
    minus_val=minus_val,
    plus_val=minus_val,
    curr_ax=curr_axis_wave
) for x in sig_vals_dlan]
xvals_sj = [plot.get_xval_dates(
    x,
    minus_val=minus_val,
    plus_val=minus_val,
    curr_ax=curr_axis_wave
) for x in sig_vals_sj]

for xval_pair in xvals_dlan:
    xval_min = xval_pair[0]
    xval_max = xval_pair[1]
    curr_axis_wave.axhline(
        ycoord_dlan_wave,
        xmin=xval_min,
        xmax=xval_max,
        color="C1"
    )
for xval_pair in xvals_sj:
    xval_min = xval_pair[0]
    xval_max = xval_pair[1]
    curr_axis_wave.axhline(
        ycoord_sj_wave,
        xmin=xval_min,
        xmax=xval_max,
        color='C2'
    )
    
   
curr_axis_wave.fill_between(
    dark_index,
    500,
    alpha=lights_alpha
)

# Tidy axes and labels
curr_axis_wave.xaxis.set_major_formatter(xfmt)
curr_axis_wave.set_ylim([0, 30])
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

for condition in condition_order:
    curr_data_total = tot_act_days.loc[condition]
    curr_data_total = curr_data_total.iloc[1:, :-1].copy()
    curr_data_total.index = curr_data_total.index.droplevel(1)
    curr_data_total = curr_data_total.T
    curr_tot_mean = curr_data_total.mean(axis=1)
    curr_tot_sem = curr_data_total.sem(axis=1)
    
    curr_axis_total.errorbar(
        curr_data_total.index,
        curr_tot_mean,
        yerr=curr_tot_sem,
        capsize=capsize_totals
    )

# Tidy axes
# curr_axis_total.set_xlim(
#     curr_data_total.index[0] - pd.Timedelta('1D'),
#     curr_data_total.index[-1] + pd.Timedelta("1D")
# )
total_xticks = curr_data_total.index[::4]
total_xticklabels = [(x - total_xticks[0]) for x in total_xticks]
curr_axis_total.set_xticks(total_xticks)
curr_axis_total.set_xticklabels(total_xticklabels)
curr_axis_total.ticklabel_format(
    axis='y',
    scilimits=(0, 0)
)



# add in stats
# ycoords
ycoord_dlan_tot = plot.sig_line_coord_get(curr_axis_total, yleveldlan )
ycoord_sj_tot = plot.sig_line_coord_get(curr_axis_total, ylevelsj)

# xcoords from marker
sig_vals_dlan = plot.sig_locs_get(
    tot_posthoc,
    index_level2val=0,
    index_levelget=0
)
sig_vals_sj = plot.sig_locs_get(
    tot_posthoc,
    index_level2val=2,
    index_levelget=0
)

# plot xvals
minus_val = 0.5
no_days = tot_act_days.columns[-1]
label_loc_total = dict(zip(range(no_days), range(no_days)))
plot.draw_sighlines(
    yval=ycoord_dlan_tot,
    sig_list=sig_vals_dlan,
    label_loc_dict=label_loc_total,
    minus_val=minus_val,
    plus_val=minus_val,
    color='C1',
    curr_ax=curr_axis_total
)
plot.draw_sighlines(
    yval=ycoord_sj_tot,
    sig_list=sig_vals_sj,
    label_loc_dict=label_loc_total,
    minus_val=minus_val,
    plus_val=minus_val,
    color='C2',
    curr_ax=curr_axis_total
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
    "Singly Housed Circadian Parameters DD"
)

fig.set_size_inches(11.69, 8.27)

plt.savefig(save_fig, dpi=600)

plt.close('all')