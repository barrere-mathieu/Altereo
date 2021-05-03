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
PATH = "../data/"

# Chargement dataset
df = pd.read_csv(PATH + "master_df_events.csv")
df_all = pd.read_csv(PATH + 'master_df_all.csv')

df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

# préparation des données
df['DDP'] = pd.to_datetime(df['DDP'])
df_all['DDP'] = pd.to_datetime(df_all['DDP'])

df['year_pose'] = df['DDP'].apply(lambda x: x.year)
df_all['year_pose'] = df_all['DDP'].apply(lambda x: x.year)

group = df.groupby(['MATERIAU', 'year_pose']).size().reset_index()
group_all = df_all.groupby(['MATERIAU', 'year_pose']).size().reset_index()
group_all = group_all.merge(group, on=["year_pose", "MATERIAU"])

# Graph date de pose avec bar chart
# fonction pour les couleurs
clrs = []
for mat in group_all['MATERIAU']:
    if mat == "ACIER":
        clrs.append("rgb(90, 44, 158, 1.0)")
    elif mat == "BONNA":
        clrs.append("rgb(128, 128, 128, 1.0)")
    elif mat == "FONTEDUCTILE":
        clrs.append("rgb(0, 0, 0, 1.0)")
    elif mat == "FONTEGRISE":
        clrs.append("rgb(42, 227, 210, 1.0)")
    elif mat =="PEBD":
        clrs.append( "rgb(42, 227, 39, 1.0)")
    elif mat == "PEHD":
        clrs.append("rgb(223, 227, 39, 1.0)")
    elif mat == "PLOMB":
        clrs.append("rgb(223, 0, 0, 1.0)")
    else:
        clrs.append("rgb(9, 0, 255, 1.0)")

P = [(x/y)*100 for x, y in zip(list(group_all["0_y"]),list(group_all["0_x"]))]
group_all["pourcentage"] = P
group_all["Couleur"] =clrs
list_mat = group_all['MATERIAU'].unique()
#new_data = pd.DataFrame({"materiau":pd.Series(m),"annee": pd.Series(annee),"pourcentage":pd.Series(P), "couleur":pd.Series(clrs)})

data = []
m = list()
annee = list()
for mat in list_mat:
    print(mat)
    print(list(group_all[group_all.MATERIAU == mat].year_pose))
    print(list(group_all[group_all.MATERIAU == mat].pourcentage))
    trace1 = go.Bar(
        name = mat,
        x = list(group_all[group_all.MATERIAU == mat].year_pose),
        y = list(group_all[group_all.MATERIAU == mat].pourcentage),
        marker = dict(color = group_all[group_all.MATERIAU == mat].Couleur,
                       line = dict(color ='rgba(255, 174, 255, 0.5)',width =0.5)),
        text =mat)
    data.append(trace1)

layout = go.Layout(title='Pourcentage de casse vs. année de pose',
                    xaxis=dict(title='Année de pose'), yaxis=dict(title = "Pourcentage de casse")
                    )

fig = go.Figure(data = data, layout = layout)
fig.update_layout(barmode='stack')

fig.show()
######### pourcentage de casses en fonction du diamètre
df = df.drop(df[df.DIAMETRE == 0].index)
df_all = df_all.drop(df_all[df_all.DIAMETRE == 0].index)
group = df.groupby(['MATERIAU', 'DIAMETRE']).size().reset_index()
group_all = df_all.groupby(['MATERIAU', 'DIAMETRE']).size().reset_index()

group_all = group_all.merge(group, on = ['MATERIAU', 'DIAMETRE'])

### bar chart
# on va récupérer juste les données pour Diamètre <= 20000 mm
group_all = group_all[group_all.DIAMETRE <= 20000]

clrs = []
for mat in group_all['MATERIAU']:
    if mat == "ACIER":
        clrs.append("rgb(90, 44, 158, 1.0)")
    elif mat == "BONNA":
        clrs.append("rgb(128, 128, 128, 1.0)")
    elif mat == "FONTEDUCTILE":
        clrs.append("rgb(0, 0, 0, 1.0)")
    elif mat == "FONTEGRISE":
        clrs.append("rgb(42, 227, 210, 1.0)")
    elif mat =="PEBD":
        clrs.append( "rgb(42, 227, 39, 1.0)")
    elif mat == "PEHD":
        clrs.append("rgb(223, 227, 39, 1.0)")
    elif mat == "PLOMB":
        clrs.append("rgb(223, 0, 0, 1.0)")
    else:
        clrs.append("rgb(9, 0, 255, 1.0)")

P = [(x / y) * 100 for x, y in zip(list(group_all["0_y"]), list(group_all["0_x"]))]
print(P)
group_all["pourcentage"] = P
group_all["Couleur"] = clrs
list_mat = group_all['MATERIAU'].unique()
data = []
for mat in list_mat:
    print(mat)
    print(list(group_all[group_all.MATERIAU == mat].DIAMETRE))
    print(list(group_all[group_all.MATERIAU == mat].pourcentage))
    trace1 = go.Bar(
        name=mat,
        x=list(group_all[group_all.MATERIAU == mat].DIAMETRE),
        y=list(group_all[group_all.MATERIAU == mat].pourcentage),
        marker=dict(color=group_all[group_all.MATERIAU == mat].Couleur,
                    line=dict(color='rgba(255, 174, 255, 0.5)', width=0.5)),
        text=mat,
        width=50
    )
    data.append(trace1)

layout = go.Layout(title='Pourcentage de casse vs. année de pose',
                   xaxis=dict(title='Diamètre du pipe'), yaxis=dict(title="Pourcentage de casses")
                   )

fig = go.Figure(data=data, layout=layout)
fig.update_layout(barmode='stack')

fig.show()