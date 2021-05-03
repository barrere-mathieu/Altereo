import pandas as pd
import numpy as np
from itertools import combinations
from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go
from plotly import subplots
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# Path to data
PATH = "data/"

df = pd.read_csv("data/master_df_events.csv")

df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)


# nombre de casses en fonction date de pose
df['DDP'] = pd.to_datetime(df['DDP'])
df['year_pose'] = df['DDP'].apply(lambda x: x.year)
df['year_pose_range'] = pd.cut(df['year_pose'], 10).astype(str)
group = df.groupby(['MATERIAU', 'year_pose_range']).size()

years = group.index.levels[1]
data = []

for mat in group.index.levels[0]:
    print(mat)
    print(list(group[mat]))
    print(group[mat].index)
    trace = go.Scatter(
                        y = list(group[mat]),
                        x = group[mat].index,
                        mode = "lines",
                        name = mat,
                        text = list(group[mat]))
    data.append(trace)

layout = go.Layout(title='Nombre de casse vs. année de pose et matériaux',
                   xaxis=dict(title='Année de pose'), yaxis=dict(title = "Nombre de casses")
                   )
fig = go.Figure(data=data, layout=layout)
fig.show()

# nombre de casses en fonction du diamètre
df = df.drop(df[df.DIAMETRE == 0].index)
df['diametre_range'] = pd.cut(df['DIAMETRE'], 10).astype(str)
group = df.groupby(['MATERIAU', 'diametre_range']).size()

diametre = group.index.levels[1]
data = []
for mat in group.index.levels[0]:
    print(mat)
    print(list(group[mat]))
    print(group[mat].index)
    trace = go.Scatter(
                        y = list(group[mat]),
                        x = group[mat].index,
                        mode = "lines",
                        name = mat,
                        text = list(group[mat]))
    data.append(trace)

layout = go.Layout(title='Nombre de casse vs. diamètre et matériaux', yaxis=dict(title = "Nombre de casses"),
                   xaxis=dict(title='Diamètre'),
                   )
fig = go.Figure(data=data, layout=layout)
fig.show()

###############
group = df.groupby(['diametre_range', 'year_pose_range']).size().reset_index()

for elt in group:
    nb_casses = group[elt]

fig = px.scatter(group, x="diametre_range", y="year_pose_range", size=nb_casses,
                 size_max=40, color=nb_casses)

fig.show()