# Figure 3. Episodes. Trying to decide whether population of episodes is
# different between the conditions

### Imports
import pathlib
import pandas as pd

idx = pd.IndexSlice
import numpy as np
import pingouin as pg
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
save_fig = pathlib.Path(
    "/Users/angusfisk/Documents/"
    "01_PhD_files/01_projects/01_thesisdata/04_ageing/"
    "03_analysisoutputs/01_figures/03_fig3.png"
)
col_names = ["Condition", "Day", "Animal", "Measure"]


def longform(data, col_names):
    new_data = data.stack().reset_index()
    new_data.columns = col_names
    return new_data


### Step 1 Read in data
file_dir = pathlib.Path(
    "/Users/angusfisk/Documents/01_PhD_files/01_projects"
    "/01_thesisdata/04_ageing"
    "/01_datafiles/01_activity/02_episodes"
)

files = sorted(file_dir.glob("*.csv"))
file_names = [x.stem[:-5].upper() for x in files]
df_list = [prep.read_file_to_df(x, index_col=0) for x in files]

# hack to remove SJ PIR 1 and 2
sj_df = df_list[-1]
sj_df.iloc[:, 0:2] = np.nan
df_list[-1] = sj_df

# gather into a df
df_dict = dict(zip(file_names, df_list))
all_df = pd.concat(df_dict, sort=False)
# all_df.drop("LDR", axis=1, inplace=True)

# find lights on
all_lights_on = all_df[all_df.loc[:, "LDR"] > 100]
dlan_lights_on = all_lights_on.loc["DLAN"].first_valid_index()
sj_lights_on = all_lights_on.loc["SJ"].first_valid_index()
ld_lights_on = all_lights_on.loc["LD"].first_valid_index()
# shift start to the first lights on
exp_data = all_df.loc[
    idx[:, dlan_lights_on.round("T"):"2019"],
    :"PIR6"
]

# import the LD and DD singly housed data as well
single_dir = file_dir.parent / "03_single_housed/02_episodes"
files_single = sorted(single_dir.glob("*.csv"))
names_single = [x.stem.upper() for x in files_single]
df_list_single = [
    prep.read_file_to_df(x, index_col=[0, 1]) for x in files_single
]
df_dict_single = dict(zip(names_single, df_list_single))
all_df_single = pd.concat(df_dict_single, sort=False)

# PIRS 6-11 look like they had LL for a few weeks from the actograms
# No notes to support what actually happened but unsure so excluding
all_df_single.loc[idx["LD"], "7":"12"] = np.nan
all_df_single.loc[idx["SJ"], "10"] = np.nan

# Split into LD and DD portions
exp_data_ld = all_df_single.loc[idx[:, "ld", :],  :"12"]
exp_data_ld.index = exp_data_ld.index.droplevel(1)
exp_data_dd = all_df_single.loc[idx[:, "dd", :], :"12"]
exp_data_dd.index = exp_data_dd.index.droplevel(1)

# remove first 7 days of DD and PIR "8"
exp_data_dd = exp_data_dd.loc[
    idx[:, (exp_data_dd.index.get_level_values(1)[0] + pd.Timedelta("14D")):],
    :
]
exp_data_dd.loc[idx["SJ"], ["8", "10"]] = np.nan

# manual mean groupby currently uses a three level multi-index
# going to artificially turn DD into higher multi-index rather
# than rewrite
dd_dict = {"level_1": exp_data_dd}
dd_df_1 = pd.concat(dd_dict)
dd_dict_2 = {"level_2": dd_df_1}
dd_df_2 = pd.concat(dd_dict_2)
dd_df_3 = dd_df_2.stack().reorder_levels([0, 1, 2, 4, 3])

all_data_dict = {
    "Group_housed": exp_data,
    "Single LD": exp_data_ld,
    "Single DD": exp_data_dd
}

periods_file = pathlib.Path(
    "/Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/"
    "04_ageing/03_analysisoutputs/01_figures/00_csvs/05_fig5/01_ddperiods.csv"
)
period_df = pd.read_csv(periods_file, index_col=[0, 1], parse_dates=True)
period_df_timedelta = pd.to_timedelta(period_df.stack()).unstack()
period_dict = {"level_2": period_df_timedelta}
period_df_1 = pd.concat(period_dict)

### Step 2 calculate length per day and count per day
# Get the number and mean duration for each day then take the mean for each
# animal
mean_data_dict = {}
count_data_dict = {}
hist_data_dict = {}
mean_cols = [col_names[0], col_names[2], col_names[3]]
mean_cols[-1] = "Mean"
count_cols = [col_names[0], col_names[2], col_names[3]]
count_cols[-1] = "Count"
bins = [(x*60) for x in [0, 1, 10, 60, 600000000]]
hist_bin_cols = ["0-1", "1-10", "10-60", ">60"]
hist_cols = [
    col_names[0],
    col_names[2],
    "Episode Duration",
    "Number of Episodes"
]



for data_type_curr, exp_data_curr in zip(all_data_dict.keys(),
                                         all_data_dict.values()):
    mean_inday_data = exp_data_curr.groupby(
        level=0
    ).resample(
        "D",
        level=1
    ).mean()
    
    bool_data = (exp_data_curr > 0).astype(bool)
    count_inday_data = bool_data.groupby(level=0).resample("D", level=1).sum()

    # print(mean_inday_data.head(), count_inday_data.head())
    if data_type_curr == list(all_data_dict.keys())[-1]: # if DD
        mean_inday_data = prep.manual_resample_mean_groupby(
            dd_df_3,
            period_df_1,
            mean=True,
            sum=False,
        )
        bool_data = (dd_df_3 > 0).astype(bool)
        count_inday_data = prep.manual_resample_mean_groupby(
            bool_data,
            period_df_1,
            mean=False,
            sum=True
        )
        mean_inday_data.index = mean_inday_data.index.droplevel([0, 1])
        mean_inday_data = mean_inday_data.unstack(level=1)
        count_inday_data.index = count_inday_data.index.droplevel([0, 1])
        count_inday_data = count_inday_data.unstack(level=1)
    
    mean_data = mean_inday_data.groupby(level=0).mean()
    count_inday_data.replace(0, np.nan, inplace=True)
    count_data = count_inday_data.groupby(level=0).mean()

    # print(mean_inday_data.head(), count_inday_data.head())
    # print(mean_data.head(), count_data.head())
    
    long_mean = longform(mean_data, mean_cols)
    long_count = longform(count_data, count_cols)

    # Calculate histogram data
    hist_input = exp_data_curr.stack().reorder_levels([0, 2, 1])
    hist_anim_data = hist_input.groupby(
        level=[0, 1]
    ).apply(
        als.hist_vals,
        bins,
        hist_bin_cols
    ).unstack()
    hist_anim_data.columns = hist_anim_data.columns.droplevel(0)

    long_hist = longform(hist_anim_data, col_names=hist_cols)

    mean_data_dict[data_type_curr] = long_mean
    count_data_dict[data_type_curr] = long_count
    hist_data_dict[data_type_curr] = long_hist
    
    
# #### Stats #####################################################################
#
# stats_colnames = ["Protocol", "Day", "Animal", "Value"]
# protocol_col = stats_colnames[0]
# day_col = stats_colnames[1]
# anim_col = stats_colnames[2]
# dep_var = stats_colnames[3]
# type_list = ["Median", "Count"]
# save_test_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
#                              "01_projects/01_thesisdata/04_ageing/"
#                              "03_analysisoutputs/01_figures/00_csvs/03_fig3")
# anova_str = "01_anova.csv"
# ph_str = "02_posthoc.csv"
#
# # run on test and count data
# for test_df, test_label in zip([mean_data, count_data], type_list):
#     print(test_label)
#     label_dir = save_test_dir / test_label
#     # tidy data
#     label_df = test_df.groupby(level=0
#                                ).apply(prep.label_anim_cols
#                                        ).stack().reset_index()
#     label_df.columns = stats_colnames
#
#     # run rm anova
#     anova = pg.mixed_anova(dv=dep_var,
#                            within=day_col,
#                            between=protocol_col,
#                            subject=anim_col,
#                            data=label_df)
#     pg.print_table(anova)
#
#     # run post hoc
#     ph_df = prep.tukey_pairwise_ph(label_df,
#                                    hour_col=day_col)
#
#     # save files
#     anova.to_csv((label_dir / anova_str))
#     ph_df.to_csv((label_dir / ph_str))
#

### Step 3 plot all ############################################################

# plotting constants
conditions = exp_data.index.get_level_values(0).unique()
condition_col = col_names[0]
day_col = col_names[1]
animal_col = col_names[2]
mean_col = mean_cols[-1]
count_col = count_cols[-1]


# Initialise the figure
fig = plt.figure()

###### Frag columns
frag_grid = gs.GridSpec(
    nrows=2,
    ncols=3,
    figure=fig,
    bottom=0.55
)
frag_axes = [plt.subplot(x) for x in frag_grid]
count_axes_dict = dict(zip(all_data_dict.keys(), frag_axes[:3]))
mean_axes_dict = dict(zip(all_data_dict.keys(), frag_axes[3:]))
frag_axes_dict = {
    count_col: count_axes_dict,
    mean_col: mean_axes_dict
}
frag_data_dict = {
    count_col: count_data_dict,
    mean_col: mean_data_dict
}

# plot count and mean duration
# Tidy constants for frag
capsize = 0.2
errwidth = 1
sem = 68
marker_size = 3

# loop through each data type
for curr_data_group in all_data_dict.keys():
    # loop through count and mean
    for curr_frag_type in frag_axes_dict.keys():
        
        curr_axis_frag = frag_axes_dict[curr_frag_type][curr_data_group]
        curr_data_frag = frag_data_dict[curr_frag_type][curr_data_group]
        
        sns.pointplot(
            x=condition_col,
            y=curr_frag_type,
            hue=condition_col,
            data=curr_data_frag,
            ax=curr_axis_frag,
            join=False,
            capsize=capsize,
            errwidth=errwidth,
            ci=sem
        )
        sns.swarmplot(
            x=condition_col,
            y=curr_frag_type,
            color='k',
            data=curr_data_frag,
            ax=curr_axis_frag,
            size=marker_size
        )
        
        curr_frag_leg = curr_axis_frag.legend()
        curr_frag_leg.remove()
        # set the title
        if curr_frag_type == list(frag_axes_dict.keys())[0]:
            curr_axis_frag.text(
                0.5,
                1.1,
                curr_data_group,
                transform=curr_axis_frag.transAxes,
                ha='center'
            )

###### Histogram plot
# create histogram axis - singular
hist_grid = gs.GridSpec(
    nrows=1,
    ncols=3,
    figure=fig,
    top=0.45
)
hist_axes = [plt.subplot(x) for x in hist_grid]

# constants for histogram plotting
duration_col = hist_cols[-2]
no_ep_col = hist_cols[-1]

dodge = 0.5

# loop through each data type for each column.
# Each condition = hue

for data_type_curr, curr_axis_hist in zip(all_data_dict.keys(), hist_axes):
    curr_data_hist = hist_data_dict[data_type_curr]

    sns.barplot(
        x=duration_col,
        y=no_ep_col,
        hue=condition_col,
        data=curr_data_hist,
        ax=curr_axis_hist,
        capsize=capsize,
        errwidth=errwidth,
        ci=sem,
        dodge=dodge
    )
    sns.swarmplot(
        x=duration_col,
        y=no_ep_col,
        hue=condition_col,
        color='k',
        data=curr_data_hist,
        ax=curr_axis_hist,
        size=marker_size,
        dodge=dodge
    )
    
    curr_leg_hist = curr_axis_hist.legend()
    curr_leg_hist.remove()
    if data_type_curr != list(all_data_dict.keys())[0]:
        curr_axis_hist.set_ylabel("")

fig.set_size_inches(11.69, 8.27)

plt.savefig(save_fig, dpi=600)

plt.close('all')
