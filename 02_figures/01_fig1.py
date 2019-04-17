# actograms of all the different conditions. Activity? sleep?
# Lets do it landscape

# imports
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.dates as mdates
import seaborn as sns
# sns.set()
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.actogram_plot as aplot

# read in the data files
file_dir = pathlib.Path(
    "/Users/angusfisk/Documents/01_PhD_files/01_projects"
    "/01_thesisdata/04_ageing"
    "/01_datafiles/01_activity"
)
save_dir = pathlib.Path(
    "/Users/angusfisk/Documents/"
    "01_PhD_files/01_projects/01_thesisdata/04_ageing/"
    "03_analysisoutputs/01_figures"
)

files = sorted(file_dir.glob("*.csv"))
file_names = [x.stem.upper() for x in files]
df_list = [prep.read_file_to_df(x, index_col=0) for x in files]
df_dict = dict(zip(file_names, df_list))
all_df = pd.concat(df_dict, sort=False)

post_file_dir = pathlib.Path(
    "/Users/angusfisk/Documents/01_PhD_files/01_projects"
    "/01_thesisdata/04_ageing"
    "/01_datafiles/01_activity/01_post_disrupt"
)
post_files = sorted(post_file_dir.glob("*.csv"))
post_file_names = [x.stem.upper() for x in post_files]
post_list = [prep.read_file_to_df(x, index_col=0) for x in files]
post_dict = dict(zip(post_file_names, post_list))
post_df = pd.concat(post_dict, sort=False)

# resample so can actually work with this data
# all_df_h = all_df.groupby(level=0).resample("H", level=1).mean()

# for each condition, plot an actogram
# doing three actograms as given amount of data want to be as big as possible
#  and the sleep actograms don't really add much at the moment

# plotting constants
conditions = all_df.index.get_level_values(0).unique()
conditions_sorted = [conditions[1], conditions[0], conditions[2]]
month_str = '2017-09'
start_day = 332
anim_dict = {
    conditions_sorted[0]: 0,
    conditions_sorted[1]: 0,
    conditions_sorted[2]: 3
}
actogram_kwargs = {
    "drop_level": False,
    "set_file_title": False,
    "linewidth": 0.1,
    "day_label_size": 8,
    "ylabelpos": (0.1, 0.5),
    "xlabelpos": (0.5, 0.1)
}

# plot actograms. Whole year with last month zoomed in at the bottom

# initialise the figure
fig = plt.figure()

# create the area for the full actograms
upper_grid = gs.GridSpec(nrows=1, ncols=len(conditions), figure=fig,
                         bottom=0.45)
upper_axes = [plt.subplot(x) for x in upper_grid]

# plot each condition in the upper axes
for condition, axis in zip(conditions_sorted, upper_axes):
    
    # select the data and the label
    data = all_df.loc[condition]
    animal = anim_dict[condition]
    
    # set titles and labels
    axis.set(yticks=[],
             xticks=[],
             title=condition)
    
    aplot._actogram_plot_from_df(
        data,
        animal,
        fig=fig,
        subplot=axis,
        timeaxis=False,
        **actogram_kwargs
    )
 
# create three lower axes for the zoomed in actograms
lower_grid = gs.GridSpec(nrows=1, ncols=len(conditions), figure=fig,
                         top=0.45)
lower_axes = [plt.subplot(x) for x in lower_grid]

# plot zoomed in conditions on the lower axes
for condition, axis in zip(conditions_sorted, lower_axes):
    
    # select the data and the label
    data = post_df.loc[condition]
    month_data = data.loc[month_str]
    animal = anim_dict[condition]
    
    # set titles and labels
    axis.set(yticks=[],
             xticks=[])
    
    aplot._actogram_plot_from_df(
        month_data,
        animal,
        fig=fig,
        subplot=axis,
        start_day=start_day,
        **actogram_kwargs
    )
 
# tidy up the figure
fig.set_size_inches(11.69, 8.27)

save_name = save_dir / "01_fig1.png"
plt.savefig(save_name, dpi=600)

# TODO add in month of september at the bottom

plt.close('all')
