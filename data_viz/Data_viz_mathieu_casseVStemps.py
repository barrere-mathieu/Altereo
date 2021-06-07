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
# Drop unknown data
df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
# Converting dates into datetime objects
df['DDCC'] = pd.to_datetime(df['DDCC'])
df['year_casse'] = df['DDCC'].apply(lambda x: x.year)
# Creation of new empty dataset with all the break dates
df_concat = pd.DataFrame({'year_casse':df.year_casse.unique()}).sort_values(by='year_casse')
df_concat.index = df_concat['year_casse']
# Preparing subplot structure
collectivite = df['collectivite'].unique()
fig = subplots.make_subplots(rows=5, cols=5, specs=[[{}, {}, {}, {}, {}],
                                                    [{}, {}, {}, {}, {}],
                                                    [{}, {}, {}, {}, {}],
                                                    [{}, {}, {}, {}, {}],
                                                    [{}, {}, {}, {}, {}]],
                             shared_xaxes=True, shared_yaxes=True,
                             vertical_spacing=0.02, horizontal_spacing=0.005,
                             subplot_titles=collectivite
                            )

j = 0
data = []
for i in range(len(collectivite)):
    # Updating index for position reference in subplot
    if i%5 == 0:
        j+=1
    # New column in dataframe df_concat: number of breaks per year for a specific collectivity "c"
    c = collectivite[i]
    df_concat[c] = df[df["collectivite"] == c].groupby(["year_casse"]).size()
    # Defining the trace
    trace = go.Scatter(
        x = list(df_concat.index),
        y = list(df_concat[c]),
        mode = "lines",
        name = c,
        # text = df.university_name
        )
    # Adding trace to subplot
    fig.add_trace(trace, col=i%5+1, row=j)
    # Saving trace
    data.append(trace)

# All collectivities on the same graph
fig2 = go.Figure(data = data)
plotly.offline.plot(fig2)
# Subplots
fig.update_layout(showlegend=False)
plotly.offline.plot(fig)


# , as_index=False