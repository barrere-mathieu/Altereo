import pandas as pd
import numpy as np
from itertools import combinations

import plotly.graph_objs as go
from plotly import subplots
import plotly.io as pio
pio.renderers.default = "browser"

# Path to data
PATH = "data/"

df = pd.read_csv(PATH + 'master_df_events.csv')
df_all = pd.read_csv(PATH + 'master_df_events.csv')
df_all.fillna(value=0)

df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df_all = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
df_all = df.drop(df[df.MATAGE == 'r'].index)

# Bubble chart: dénombrement variables qualitatives
colonnes = ['MATERIAU', 'MATAGE', 'collectivite']
for x, y in combinations(colonnes, 2):
    group1 = df[[x, y]].groupby([x, y]).size()
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
                    np.log(group1[i][j] / np.min(list(group1))) * 10
                )
                labels.append(group1[i][j])

            except:
                size.append(0)
                labels.append(0)

    trace1 = {
            'y': Y,
            'x': X,
            'mode': 'markers',
            'marker': {
                'color': size,
                'size': size,
            },
            "text": labels,
        }

    trace2 = {
        'x': list(groupx.index),
        'y': list(groupx),
        'name': 'Hist. '+str(x),
        'type': 'bar'
    }

    trace3 = {
        'y': list(groupy.index),
        'x': list(groupy),
         'name': 'Hist. ' +str(y),
        'type': 'bar',
        'orientation': 'h'
    }

    # Creating subplots
    # fig = subplots.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{}, {}]])
    # fig.append_trace(trace3, 1, 1)
    # fig.append_trace(trace1, 2, 1)
    # fig.append_trace(trace2, 2, 2)

    fig = subplots.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=False,
                                 shared_yaxes=True)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace3, 1, 2)

    # fig = go.Figure(data=[trace3])
    fig.show()


