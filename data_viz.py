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


df = pd.read_csv(PATH + 'master_df.csv')
df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)

# Bubble chart: dénombrement variables qualitatives
colonnes = ['MATERIAU', 'MATAGE', 'collectivite']
for x, y in combinations(colonnes, 2):
    data = df[[x, y]]
    group = data.groupby([x,y]).size()
    serie = np.log(list(group)/np.min(list(group)))
    # international_color = [float(each) for each in df2016.international]
    trace1 = [
        {
            'y': [group.index.levels[1][k] for k in group.index.labels[1]],
            'x': [group.index.levels[0][k] for k in group.index.labels[0]],
            'mode': 'markers',
            'marker': {
                'color': serie,
                'size': serie,
                'showscale': True,
            },
            "text" : list(group),
        }
    ]
    # iplot(data)
    fig = go.Figure(data = trace1)
    fig.show()
