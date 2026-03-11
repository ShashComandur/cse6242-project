# Before running this file, run the following below in the command prompt
#    python -m pip install pybaseball pandas pyarrow tqdm

# imports
from pybaseball import statcast
import pandas as pd
from tqdm import tqdm

# function to retrieve info from statcast
def pull_year(year: int, start="03-01", end="11-30"):
    start_dt = f"{year}-{start}"
    end_dt   = f"{year}-{end}"
    df = statcast(start_dt, end_dt)
    return df

# define years
years = [2015,2016,2017,2018,2019,2020, 2021, 2022, 2023, 2024]

# concat dataframes
dfs   = []

# loop over years
for y in tqdm(years):
    df_y = pull_year(y)
    #df_y.to_csv(f"statcast_{y}.csv", index=False)
    dfs.append(df_y)

# concat data
df_all = pd.concat(dfs,ignore_index = True)

# info about data
print(df_all.info())

# store combined file
df_all.to_csv(f"statcast_{years[0]}_{years[-1]}.csv",index=False)