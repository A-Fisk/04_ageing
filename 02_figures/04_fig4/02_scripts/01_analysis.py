# Script for doing all analysis for figure 4.

import pathlib
import pandas as pd
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/actiPy")
import actiPy.periodogram as per

# read in files ################################################################
fildir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                      "01_projects/01_thesisdata/04_ageing/"
                      "01_datafiles/01_activity/03_single_housed/"
                      "01_concat")
filenames = sorted(fildir.glob("*.csv"))
read_kwargs = {
    "index_col": [0, 1],
    'parse_dates': True
}
ld_str = "ld"
conditions = list(set([x.stem for x in filenames]))
files = [pd.read_csv(x, **read_kwargs).loc[ld_str]
         for x in filenames]
df_dict = dict(zip(conditions, files))
data = pd.concat(df_dict)

# Calculate values

# QP
ls_power = per.get_period(data, return_power=True, drop_lastcol=True)


# IS

# IV

# Mean activity

# save files
