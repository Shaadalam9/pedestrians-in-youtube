# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import pandas as pd
import glob
import shutil
import ast


def parse_nested_list(cell):
    """Converts a string like '["[0]", "[1]"]' into [[0], [1]]"""
    if pd.isna(cell):
        return []
    try:
        parsed = ast.literal_eval(cell)
        if all(isinstance(i, str) and i.startswith('[') for i in parsed):
            return [ast.literal_eval(i) for i in parsed]
        return parsed
    except Exception:
        return []


def parse_flat_list(cell):
    """Parses '["val1", "val2"]' or '[val1,val2]' into ['val1', 'val2']"""
    if pd.isna(cell):
        return []
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return [item.strip(" '\"") for item in str(cell).strip("[]").split(',') if item.strip()]


# filenames
base_file = 'mapping.csv'
backup_file = 'mapping_bkp.csv'

# create backup
shutil.copyfile(base_file, backup_file)
print(f"Backed up {base_file} -> {backup_file}")

# list-type columns
list_columns = ['videos', 'time_of_day', 'start_time', 'end_time', 'vehicle_type', 'upload_date', 'fps_list',
                'channel']
nested_list_columns = ['time_of_day', 'start_time', 'end_time']

# load main mapping.csv
main_df = pd.read_csv(base_file)
for col in list_columns:
    if col in nested_list_columns:
        main_df[col] = main_df[col].apply(parse_nested_list)
    else:
        main_df[col] = main_df[col].apply(parse_flat_list)

# process each mapping-NAME.csv
for file in glob.glob('mapping-*.csv'):
    if file == base_file or file == backup_file:
        continue
    print(f"Working with {file}")
    new_df = pd.read_csv(file)
    for col in list_columns:
        if col in nested_list_columns:
            new_df[col] = new_df[col].apply(parse_nested_list)
        else:
            new_df[col] = new_df[col].apply(parse_flat_list)

    for _, new_row in new_df.iterrows():
        city = new_row['city']
        match = main_df[main_df['city'] == city]

        if not match.empty:
            idx = match.index[0]
            existing_videos = set(main_df.at[idx, 'videos'])
            new_videos = [vid for vid in new_row['videos'] if vid not in existing_videos]

            if new_videos:
                for col in list_columns:
                    for vid_idx, vid in enumerate(new_row['videos']):
                        if vid in new_videos:
                            value = new_row[col][vid_idx]
                            main_df.at[idx, col].append(value)
        else:
            main_df = pd.concat([main_df, pd.DataFrame([new_row])], ignore_index=True)

# reindex
main_df.reset_index(drop=True, inplace=True)
main_df['id'] = main_df.index + 1


# convert lists to string format
def format_list(lst):
    return f"[{','.join(str(x) for x in lst)}]"


def format_nested_list(lst):
    inner = [f"[{','.join(str(i) for i in sublist)}]" for sublist in lst]
    return f"[{','.join(inner)}]"


for col in list_columns:
    if col in nested_list_columns:
        main_df[col] = main_df[col].apply(format_nested_list)
    else:
        main_df[col] = main_df[col].apply(format_list)

# write to mapping.csv
main_df.to_csv(base_file, index=False)
print(f"Updated {base_file} with new videos from mapping-*.csv files.")
