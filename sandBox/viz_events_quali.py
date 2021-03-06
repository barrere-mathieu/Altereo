import pandas as pd
import numpy as np
from itertools import combinations

import plotly
plotly.offline.iplot
import plotly.graph_objs as go
from plotly import subplots
import plotly.io as pio
pio.renderers.default = "browser"

# Path to data
PATH = "../data/"

df = pd.read_csv(PATH + 'master_df_events.csv')
df_all = pd.read_csv(PATH + 'master_df_all.csv')
df_all.fillna(value=0)

df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df[df.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
df_all = df_all.drop(df[df.MATAGE == 'r'].index)

df_all['DDP'] = pd.to_datetime(df_all['DDP'])
df_all['year_pose'] = df_all['DDP'].apply(lambda x: x.year)
df_all['year_pose_range'] = pd.cut(df_all['year_pose'], 10).astype(str)
# df_all['diametre_range'] = pd.cut(np.log(df_all['DIAMETRE']), 5).astype(str)
df_all['diametre_range'] = pd.qcut(df_all['DIAMETRE'], q=5).astype(str)

df= pd.merge(df, df_all[['ID', 'year_pose_range', 'diametre_range']], on='ID')
# df.to_csv(PATH + 'df_merge_test.csv', index = False)

# Bubble chart: dénombrement variables qualitatives
# colonnes = ['MATERIAU', 'MATAGE', 'collectivite', 'year_pose_range', 'diametre_range']
colonnes = ['year_pose_range', 'diametre_range', 'MATERIAU', 'MATAGE', 'collectivite']
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
                    np.log(group1[i][j] / np.min(list(group1)))*10
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
            name = 'Nombre de Casses : Histogramme 3D',
            showlegend=True
    )

    # Histogramme X
    trace2 = go.Bar(
        x = list(groupx.index),
        y = list(groupx),
        name = 'Nombre de casses par '+str(x),
        marker = dict(color='rgba(0, 0, 150, 0.6)', line=dict(color='rgba(0, 0, 150, 0.6)', width=1)),
        text=list(groupx), textposition="auto"
    )

    # Histogramme Y
    trace3 = go.Bar(
        y = list(groupy.index),
        x = list(groupy),
        name = 'Nombre de casses par ' +str(y),
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

    fig.update_layout(title_text="Histogramme du nombre de casses en fonction de "+str(x)+" et "+str(y))
    # fig.show()
    plotly.offline.plot(fig)
