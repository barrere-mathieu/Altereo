import pandas as pd
import math 

from plotly.subplots import make_subplots


import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

# Path to data
PATH = "../data/"

df = pd.read_csv(PATH + "master_df_events.csv")
df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)


######## nombre de casses en fonction date de casse
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
    L = list(group[group.collectivite == col].year_event)
    print(L)
    m =list(group[group.collectivite == col].nb_casses)
    print(m)
    
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

######## pourcentage de casses en fonction date de casse
df_all = pd.read_csv(PATH + "master_df_all.csv")
df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

group_all = df_all.groupby(['collectivite']).size().reset_index()
group_all = group_all.merge(group, on = ['collectivite'])

group_all['taux_casse'] = [(x/y)*100 for x, y in zip(list(group_all["0_y"]),list(group_all["0_x"]))]
#data = []
L = list()
m = list()
fig = make_subplots(
    rows=5, cols=5,
    shared_xaxes=True,
    shared_yaxes = True,
    subplot_titles=(list_collec))

for i, col in enumerate(list_collec):
    print(col)
    L = list(group_all[group_all.collectivite == col].year_event)
    print(L)
    m =list(group_all[group_all.collectivite == col].taux_casse)
    print(m)
    
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
fig.update_yaxes(title_text="Taux de casses", row= 3, col=1)

fig.show()

