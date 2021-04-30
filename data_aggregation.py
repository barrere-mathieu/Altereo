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
    for loc in list(file_list.keys()):
        path = ''.join(file_list[loc][0].split('/')[1:-1]) + '/'

        event = pd.read_csv([i for i in file_list[loc] if i.find('events') != -1][0], sep = ';', index_col=False)
        pipe = pd.read_csv([i for i in file_list[loc] if i.find('pipes') != -1][0], sep = ';', index_col=False)

        agg_df = event.merge(pipe, left_on='IDTC', right_on='IDT')
        agg_df['collectivite'] = loc
        agg_df['ID'] = agg_df['collectivite'] + "_" + agg_df['IDTC'].astype('str')
        agg_df = agg_df.drop(['IDTC'], axis = 1)
        agg_df.to_csv(PATH + path + '{}_agg_df.csv'.format(loc))
        if master_df.empty == True:
            master_df = agg_df
        else:
            master_df = pd.concat([master_df, agg_df], join="inner", axis = 0)
    master_df.to_csv(Path + 'master_df.csv')

create_dataset(PATH)