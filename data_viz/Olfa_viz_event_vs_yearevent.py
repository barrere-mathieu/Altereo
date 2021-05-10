import pandas as pd
import numpy as np
import math 

from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots


import plotly.graph_objs as go
from plotly import subplots
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# Path to data
PATH = "../data/"

df = pd.read_csv(PATH + "master_df_events.csv")
df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)

# nombre de casses en fonction date de casse
df['DDCC'] = pd.to_datetime(df['DDCC'])
df['year_event'] = df['DDCC'].apply(lambda x: x.year)
group = df.groupby(['collectivite', 'year_event']).size().reset_index()


list_collec = group['collectivite'].unique()
group['nb_casses'] = group[0]
data = []
L = list()
m = list()
fig = make_subplots(
    rows=5, cols=5,
    shared_xaxes=True,
    shared_yaxes = True,
    subplot_titles=(list_collec))

for i, col in enumerate(list_collec):
    print(col)
    print(i)
    L = list(group[group.collectivite == col].year_event)
    m =list(group[group.collectivite == col].nb_casses)
    
    fig.add_trace(go.Scatter(
            x = L,
            y = m,
            mode = "lines",
            name = col,
            #text = list(group[mat])
            ),
        row = math.floor(i/5)+1,
        col = i%5+1)

fig.update_xaxes(title_text="Année de casses", row = 5, col = 3 )
fig.update_yaxes(title_text="Nombre de casses", row= 3, col=1)

fig.show()

