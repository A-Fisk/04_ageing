# Script to plot figure 4

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import actiPy.actogram_plot as aplot

# read in files ################################################################
actogram_fildir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                               "01_projects/01_thesisdata/04_ageing/"
                               "01_datafiles/01_activity/03_single_housed/"
                               "01_concat")
actogram_filenames = sorted(actogram_fildir.glob("*.csv"))
read_kwargs = {
    "index_col": [0, 1],
    'parse_dates': True
}
ld_str = "ld"
conditions = list(set([x.stem for x in actogram_filenames]))
act_files = [pd.read_csv(x, **read_kwargs).loc[ld_str]
             for x in actogram_filenames]
act_dict = dict(zip(conditions, act_files))

# plotting constants ###########################################################
act_animno_dict = {
    "ld": 0,
    "dlan": 0,
    "sj": 0
}
kwargs_actogram = {
    "drop_level": False,
    "set_file_title": False,
    "linewidth": 0.1,
}


# plot data ####################################################################
fig = plt.figure()

actogram_grid = gs.GridSpec(nrows=1,
                            ncols=len(act_files),
                            figure=fig,
                            bottom=0.55)
axes_act = [plt.subplot(x) for x in actogram_grid]

# plot each condition in it's own col
for condition, axis in zip(conditions, axes_act):

    # select the data
    act_data = act_dict[condition]
    anim_no = act_animno_dict[condition]
    
    # plot on axis
    aplot._actogram_plot_from_df(act_data,
                                 anim_no,
                                 fig=fig,
                                 subplot=axis,
                                 timeaxis=True,
                                 **kwargs_actogram)
    
    # tidy axis
    axis.set_title(condition)