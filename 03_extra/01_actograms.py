# Script for creating short actograms

import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.actogram_plot as act

# define the input directoryies
activity_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                              "01_projects/01_thesisdata/04_ageing/"
                              "01_datafiles/01_activity/01_post_disrupt")

# define the save directories
activity_save = activity_dir.parents[2] / "03_analysisoutputs/02_misc"

# define the subdirectory in the save directory to create and save in
subdir_name = '01_actograms'

# define the keywords for reading in the file
init_kwargs = {
    "subdir_name": subdir_name,
    "func": (prep, "read_file_to_df"),
    "index_col": [0],
    "header": [0],
    "input_directory": activity_dir,
    "save_directory": activity_save
}

# define the keywords to plot the file
plot_kwargs = {
    "function": (act, "actogram_plot_all_cols"),
    "LDR": -1,
    "set_file_title": True,
    "showfig": False,
    "period": "24H",
    "savefig": True,
    "figsize": (10, 10),
    "drop_level": False,
}


short_act_object = prep.SaveObjectPipeline(**init_kwargs)
short_act_object.create_plot(**plot_kwargs)

