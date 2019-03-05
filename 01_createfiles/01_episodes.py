# Script for creating episode files

import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.episodes as ep
import actiPy.actogram_plot as act

# define the input directoryies
activity_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                              "01_projects/01_thesisdata/04_ageing"
                              "/01_datafiles/01_activity/01_post_disrupt")

# define the save directories
activity_save = activity_dir.parent

# define the subdirectory in the save directory to create and save in
subdir_name = '02_episodes'

# define the keywords for reading in the file
init_kwargs = {
    "subdir_name": subdir_name,
    "func": (prep, "read_file_to_df"),
    "index_col": [0],
    "header": [0]
}

# define the keywords to process the file
process_kwargs = {
    "function": (ep, "episode_find_df"),
    "savecsv": True,
    "drop_level": False,
    "max_time": "14H"
}

# copy the kwargs
curr_init = init_kwargs
curr_process = process_kwargs

# modify the init and plot kwargs if necessary
curr_init["input_directory"] = activity_dir
curr_init["save_directory"] = activity_save

# process all the files
ep_object = prep.SaveObjectPipeline(**curr_init)
ep_object.process_file(**curr_process)
