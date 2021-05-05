import pandas as pd
import numpy as np
from itertools import combinations

import plotly.graph_objs as go
import plotly.express as px
from plotly import subplots
import plotly.io as pio
pio.renderers.default = "browser"

# Path to data
PATH = "../data/"

df = pd.read_csv(PATH + 'master_df_events.csv')
df_all = pd.read_csv(PATH + 'master_df_all.csv')

df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

#### Graphe pourcentage de casse pour chaque matériau
group = df.groupby(['MATERIAU']).size().reset_index()
group_all = df_all.groupby(['MATERIAU']).size().reset_index()
group_all = group_all.merge(group, on=[ "MATERIAU"])

group_all["Pourcentage"] = [(x/y)*100 for x, y in zip(list(group_all["0_y"]),list(group_all["0_x"]))]


fig = px.scatter(group_all, x="MATERIAU", y=group_all["Pourcentage"], size=group_all["Pourcentage"],
                 size_max=40, color=group_all["Pourcentage"])

fig.show()




# Bubble chart: dénombrement variables qualitatives
colonnes = ['MATERIAU', 'MATAGE', 'collectivite']
for x, y in combinations(colonnes, 2):
    group1 = df[[x, y]].groupby([x, y]).size()
    group_all = df_all[[x, y]].groupby([x, y]).size()
    groupx = df_all[[x, y]].groupby([x]).size()
    groupy = df_all[[x, y]].groupby([y]).size()
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
                    group1[i][j] / group_all[i][j]*1000
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
    fig.show()
