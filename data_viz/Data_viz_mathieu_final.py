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

df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

# Découpage des années : Aggrégation années < 1930 car très peu de points
df_all['DDP'] = pd.to_datetime(df_all['DDP'])
df_all['year_pose'] = df_all['DDP'].apply(lambda x: x.year)
df_all['year_pose']=df_all['year_pose'].apply(lambda x: 1930 if x <1930 else x)
df_all['year_pose_range'] = pd.cut(df_all['year_pose'], 5).astype(str)

# Aggrégation des diamètres: on utilise la méthode qcut qui essaye de garder des quantités équivalentes dans les découpages
# df_all['diametre_range'] = pd.cut(np.log(df_all['DIAMETRE']), 5).astype(str)
df_all['diametre_range'] = pd.qcut(df_all['DIAMETRE'], q=5).astype(str)

df= pd.merge(df, df_all[['ID', 'year_pose_range', 'diametre_range']], on='ID')

# Bubble chart: dénombrement variables qualitatives
colonnes = ['year_pose_range', 'diametre_range', 'MATERIAU', 'MATAGE', 'collectivite']
for x, y in combinations(colonnes, 2):
    group1 = df[[x, y]].groupby([x, y]).size()
    group_all = df_all[[x, y]].groupby([x, y]).size()
    groupx = df_all[[x, y]].groupby([x]).size()
    groupx_ev = df[[x, y]].groupby([x]).size()
    groupy = df_all[[x, y]].groupby([y]).size()
    groupy_ev = df[[x, y]].groupby([y]).size()
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

    trace2_bis =  go.Scatter(
        x = list(groupx.index),
        y = list(groupx_ev/groupx*1000),
        mode = "lines",
        name = "% Casse (x10)",
        marker = dict(color = 'rgba(0, 150, 30, .8)'),
        # text = df.university_name
    )

    # Histogramme Y
    trace3 = go.Bar(
        y = list(groupy.index),
        x = list(groupy),
        name = 'Nombre de casses par ' +str(y),
        marker = dict(color='rgba(0, 0, 150, 0.6)', line=dict(color='rgba(0, 0, 150, 0.6)', width=1)),
        orientation= 'h',
        xaxis='x4',
        text=list(groupy), textposition="auto"
    )

    trace3_bis =  go.Scatter(
        y = list(groupy.index),
        x = list(groupy_ev/groupy*np.mean(list(groupy))/np.mean(list(groupy_ev/groupy))),
        mode = "lines",
        name = "% Casse (x10)",
        marker = dict(color = 'rgba(0, 150, 30, .8)'),
        # orientation='h',
        # text = df.university_name
        xaxis= 'x5',
    )

    fig = subplots.make_subplots(rows=2, cols=2, specs=[[{"secondary_y": True}, {"secondary_y": True}],
                                                        [{"secondary_y": False}, {"secondary_y": False}]],
                                 shared_xaxes=True,
                                 shared_yaxes=True, vertical_spacing=0.01, horizontal_spacing=0.01,
                                 column_widths=[0.8, 0.2], row_width=[0.8, 0.2],
                                 )
    fig.update_layout(xaxis5=go.layout.XAxis(overlaying='x4', side='top'))

    # Top left
    fig.add_trace(trace2, row=1, col=1, secondary_y=False)
    fig.add_trace(trace2_bis, row=1, col=1, secondary_y=True)
    # Top right EMPTY
    # fig.add_trace(trace_dummy, row=1, col=2, secondary_y=True)
    # Bottom left
    fig.add_trace(trace1, row=2, col=1)
    # Bottom righ
    fig.add_trace(trace3, row=2, col=2, secondary_y=False)
    fig.add_trace(trace3_bis, row=2, col=2)

    # fig.append_trace(trace1, 2, 1)
    # fig.append_trace(trace3, 2, 2)
    # fig.append_trace(trace2, 1, 1)
    # fig.append_trace(trace2_bis, 1, 1)

    fig.update_layout(title_text="Histogramme du nombre de casses en fonction de "+str(x)+" et "+str(y))

    # x5 = go.layout.XAxis(overlaying='x4', side='top')

    # fig.show()
    plotly.offline.plot(fig)


#
# data = [trace3, trace3_bis]
# layout = {
#   'yaxis': {'title': 'yaxis title'},
#   'xaxis2': {
#         'overlaying': 'x',
#         'side': 'top'
#   }
# }
#
# fig2 = go.Figure(data=data, layout=layout)
# plotly.offline.plot(fig2)
