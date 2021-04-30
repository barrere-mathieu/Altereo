import pandas as pd
import numpy as np
from itertools import combinations

import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"


# Path to data
PATH = "data/"


df = pd.read_csv(PATH + 'master_df_events.csv')
df_all = pd.read_csv(PATH + 'master_df_events.csv')
df_all.fillna(value = 0)

df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df_all = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
df_all = df.drop(df[df.MATAGE == 'r'].index)

# Bubble chart: dénombrement variables qualitatives
colonnes = ['MATERIAU', 'MATAGE', 'collectivite']
for x, y in combinations(colonnes, 2):
    group1 = df[[x, y]].groupby([x,y]).size()
    groupx = df[[x, y]].groupby([x]).size()
    groupy = df[[x, y]].groupby([y]).size()
    x_list = df_all[x].unique()
    y_list = df_all[y].unique()
    x_list.sort()
    y_list.sort()
    X = []
    Y = []
    size = []
    labels = []
    for i in x_list:
        for j in y_list:
            X.append(i)
            Y.append(j)
            try:
                size.append(
                    np.log(group1[i][j]/np.min(list(group1)))*10
                )
                labels.append(group1[i][j])

            except:
                size.append(0)
                labels.append(0)

    trace1 = [
        {
            'y': Y,
            'x': X,
            'mode': 'markers',
            'marker': {
                'color': size,
                'size': size,
            },
            "text" : labels,
        }
    ]

    # iplot(data)
    fig = go.Figure(data = trace1)
    fig.show()
