import pandas as pd
import numpy as np
import os
from itertools import combinations

import plotly as py
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

# Path to data
PATH = "data/"

def create_dataset(Path):
    file_list = {}
    for (path, dirs, files) in os.walk(PATH):
        l = []
        for file in files:
            location = path.split("\\")[-1]
            l.append(path+'/'+file)
        if len(files) != 0 and path.find('VillesClean_ETP') != -1:
            file_list[location] = l

    master_df = pd.DataFrame()
    master_df_all = pd.DataFrame()
    for loc in list(file_list.keys()):
        path = ''.join(file_list[loc][0].split('/')[1:-1]) + '/'
        # Read the two files
        event = pd.read_csv([i for i in file_list[loc] if i.find('events') != -1][0], sep = ';', index_col=False)
        pipe = pd.read_csv([i for i in file_list[loc] if i.find('pipes') != -1][0], sep = ';', index_col=False)
        # Merge dataset (agg_df = only events ; agg_df_all = all pipes)
        agg_df = event.merge(pipe, left_on='IDTC', right_on='IDT')
        agg_df_all = event.merge(pipe, left_on='IDTC', right_on='IDT', how='outer')
        # Add collectivité column
        agg_df['collectivite'] = loc
        agg_df_all['collectivite'] = loc
        # Add ID column (collectivité + IDT)
        agg_df['ID'] = agg_df['collectivite'] + "_" + agg_df['IDT'].astype('str')
        agg_df_all['ID'] = agg_df_all['collectivite'] + "_" + agg_df_all['IDT'].astype('str')
        # Remove redonduncies (IDTC = IDT)
        agg_df = agg_df.drop(['IDTC'], axis = 1)
        agg_df_all = agg_df_all.drop(['IDTC'], axis = 1)
        # Save dataframe
        agg_df.to_csv(PATH + path + '{}_agg_df.csv'.format(loc))
        if master_df.empty == True:
            master_df = agg_df
            master_df_all = agg_df_all
        else:
            master_df = pd.concat([master_df, agg_df], join="inner", axis = 0)
            master_df_all = pd.concat([master_df_all, agg_df_all], join="inner", axis = 0)
    # Save dataframe: only pipes with events
    master_df.to_csv(Path + 'master_df_events.csv')
    # Save dataframe: all pipes
    master_df_all.to_csv(Path + 'master_df_all.csv')

create_dataset(PATH)