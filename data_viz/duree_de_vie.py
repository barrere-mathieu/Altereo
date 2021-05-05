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

df = pd.read_csv(PATH + 'master_df_events.csv')
df_all = pd.read_csv(PATH + 'master_df_all.csv')

df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

# Découpage des années : Aggrégation années < 1930 car très peu de points
df['DDP'] = pd.to_datetime(df['DDP'])
df['year_pose'] = df['DDP'].apply(lambda x: x.year)

df['DDCC'] = pd.to_datetime(df['DDCC'])
df['year_casse'] = df['DDCC'].apply(lambda x: x.year)
# Calcul de la durée de vie
df['duree_de_vie'] = df['year_casse'] - df['year_pose']

# Aggregation data
df_all['diametre_range'] = pd.qcut(df_all['DIAMETRE'], q=5).astype(str)
df= pd.merge(df, df_all[['ID', 'diametre_range']], on='ID')

colonnes = ['diametre_range', 'MATERIAU', 'MATAGE', 'collectivite']
row = 'duree_de_vie'
for col in colonnes:
    group_duree = df[[row, col]].groupby([col]).median()
    group_size = df_all[[col]].groupby([col]).size()
    group_size_ev = df[[col]].groupby([col]).size()

    trace = go.Bar(
        x = list(group_duree.index),
        y = list(group_duree.iloc[:, 0]),
        name = 'Durée de vie',
        marker = dict(color='rgba(0, 0, 150, 0.6)', line=dict(color='rgba(0, 0, 150, 0.6)', width=1)),
        text=list(group_duree), textposition="auto"
    )

    trace2 =  go.Scatter(
        x = list(group_size.index),
        y = list(group_size_ev/group_size),
        mode = "lines",
        name = "% Casse",
        marker = dict(color = 'rgba(0, 200, 50, 1)'),
        # orientation='h',
        # text = df.university_name
        yaxis= 'y2',
    )

    layout = {
        'xaxis': {'title': col},
        'yaxis': {'title': row+' (années)'},
        'yaxis2': {
            'title': '% casse',
            'overlaying': 'y',
            'side': 'right'
        }
    }

    fig = go.Figure(data = [trace, trace2], layout=layout)
    plotly.offline.plot(fig)
