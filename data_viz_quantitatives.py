import pandas as pd
import numpy as np
from itertools import combinations

import plotly.graph_objs as go
from plotly import subplots
import plotly.io as pio
pio.renderers.default = "browser"

# Path to data
PATH = "data/"

# Chargement dataset
df = pd.read_csv(PATH + 'master_df_events.csv')
df = df.drop(df[df.MATERIAU == 'INCONNU'].index)

# Graph date de pose
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
                        y = np.log(list(group[mat])),
                        x = years,
                        mode = "lines",
                        name = mat,
                        text = list(group[mat]))
    data.append(trace)

layout = go.Layout(title='Nombre de casse vs. année de pose et matériaux',
                   xaxis=dict(title='Année de pose'),
                   )
fig = go.Figure(data=data, layout=layout)
fig.show()
