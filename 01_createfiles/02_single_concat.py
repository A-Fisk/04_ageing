# Grab all the single housed data and concat

import pandas as pd
import pathlib


input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                         "01_projects/01_thesisdata/04_ageing/01_datafiles/"
                         "01_activity/03_single_housed")

csv_list = sorted(input_dir.glob("*.csv"))
protocol_list = [x.stem[:-3] for x in csv_list]
protocol_names = list(set(protocol_list))

read_kwargs = {
    "index_col": [-1, 0],
    "parse_dates": True
}

for file_str in protocol_names:
    # mix the two files for each condition together
    file_list = sorted(input_dir.glob("%s*.csv"%(file_str)))
    dfs = [pd.read_csv(x, **read_kwargs) for x in file_list]
    old_cols = dfs[0].columns
    for col_nos, df in zip([range(1,7), range(7,13)], dfs):
        new_cols = [x for x in col_nos]
        new_cols.append(old_cols[-1])
        df.columns = new_cols
    grouped_df = pd.concat(dfs)

    # resample index to account for slihgt variation in times between records
    sorted_df = grouped_df.groupby(level=0
                                   ).resample("10S", level=1
                                   ).mean().sort_index(level=1)
    # resampling puts in na values, backfill them
    sorted_df.fillna(method="bfill", inplace=True)
    # put LDR as the final col
    ldr_col = sorted_df.pop("LDR")
    sorted_df["LDR"] = ldr_col

    # save
    sorted_dir = input_dir / "01_concat"
    save_name = sorted_dir / (file_str + ".csv")
    sorted_df.to_csv(save_name)

