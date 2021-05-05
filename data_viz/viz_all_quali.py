import pandas as pd
import numpy as np
from itertools import combinations

import plotly
import plotly.graph_objs as go
from plotly import subplots
import plotly.io as pio
pio.renderers.default = "browser"

# Path to data
PATH = "../data/"

df = pd.read_csv(PATH + 'master_df_all.csv')

df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
df['DDP'] = pd.to_datetime(df['DDP'])
df['year_pose'] = df['DDP'].apply(lambda x: x.year)
df['year_pose_range'] = pd.cut(df['year_pose'], 10).astype(str)
df['diametre_range'] = pd.cut(np.log(df['DIAMETRE']), 5).astype(str)
# df['diametre_range2'] = pd.qcut(df['DIAMETRE'], q=5).astype(str)
df['diametre_range'] = pd.qcut(df['DIAMETRE'], q=5).astype(str)


# Bubble chart: dénombrement variables qualitatives
colonnes = ['year_pose_range', 'diametre_range', 'MATERIAU', 'MATAGE', 'collectivite']
df = df[colonnes]
for x, y in combinations(colonnes, 2):
    group1 = df[[x, y]].groupby([x, y]).size()
    groupx = df[[x, y]].groupby([x]).size()
    groupy = df[[x, y]].groupby([y]).size()
    x_list = df[x].unique()
    y_list = df[y].unique()
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
                    np.log(group1[i][j] / np.min(list(group1))) * 7
                )
                labels.append(group1[i][j])

            except:
                size.append(0)
                labels.append(0)

    trace1 = go.Scatter(
            y = Y,
            x = X,
            mode = 'markers',
            marker = {
                'color': size,
                'size': size,
            },
            text =  labels,
            name = 'Nombre de tuyaux installés : Histogramme 3D',
            showlegend=True
    )

    # Histogramme X
    trace2 = go.Bar(
        x = list(groupx.index),
        y = list(groupx),
        name = 'Nombre de tuyaux installés par '+str(x),
        marker = dict(color='rgba(0, 0, 150, 0.6)', line=dict(color='rgba(0, 0, 150, 0.6)', width=1)),
        text=list(groupx), textposition="auto"
    )

    # Histogramme Y
    trace3 = go.Bar(
        y = list(groupy.index),
        x = list(groupy),
        name = 'Nombre de tuyaux installés par ' +str(y),
        marker = dict(color='rgba(0, 0, 150, 0.6)', line=dict(color='rgba(0, 0, 150, 0.6)', width=1)),
        orientation= 'h',
        text=list(groupy), textposition="auto"
    )

    fig = subplots.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{}, {}]], shared_xaxes=True,
                                 shared_yaxes=True, vertical_spacing=0.01, horizontal_spacing=0.01,
                                 column_widths=[0.8, 0.2], row_width=[0.8, 0.2],
                                 )
    fig.append_trace(trace1, 2, 1)
    fig.append_trace(trace3, 2, 2)
    fig.append_trace(trace2, 1, 1)

    fig.update_layout(title_text="Histogramme du nombre de tuyaux installés en fonction de "+str(x)+" et "+str(y))
    # fig.show()
    plotly.offline.plot(fig)
