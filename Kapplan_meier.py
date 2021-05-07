import pandas as pd
import numpy as np
import math 
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter



import plotly.graph_objs as go
from plotly import subplots
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# Path to data
PATH = "data/"

# Chargement dataset
df = pd.read_csv(PATH + "master_df_events.csv")
df_all = pd.read_csv(PATH + 'master_df_all.csv')

df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)


# préparation des données
df_all['DDP'] = pd.to_datetime(df_all['DDP'])
df_all['DDCC'] = pd.to_datetime(df_all['DDCC'])

df_all['year_pose'] = df_all['DDP'].apply(lambda x: x.year)
df_all['year_event'] = df_all['DDCC'].apply(lambda x: x.year)
df_all = df_all.drop(["DDP", "DDCC", "ID" ], axis = 1)


# Analyse de survie independemment des collectivités
df_all["duration"] = df_all['year_event'] - df_all["year_pose"]
df_all["observation"] = 1  # 0 pour tuyaux pas cassé
# remplacer l'état des tuyaux cassé de 1 --> 0
df_all.loc[df_all.year_event.isna() == True, "observation"] = 0
# remplacer la date de la casse par 2013 la dernière date d'observation
df_all.loc[df_all.year_event.isna() == True, "duration"] = 2013 - df_all["year_pose"]

kmf = KaplanMeierFitter()

T = df_all["duration"]
E = df_all["observation"]

kmf.fit(T, event_observed=E, label = "toute les collectivités")
print(kmf.survival_function_)

kmf.plot()

# récupérer toutes les données de la collectivité 22
group_all = df_all[df_all.collectivite == "Collectivite_22"]

group_all["duration"] = group_all['year_event'] - group_all["year_pose"]
a= group_all['year_event'].values
group_all["observation"] = 1  # 0 pour tuyaux pas cassé
# remplacer l'état des tuyaux cassé de 1 --> 0
group_all.loc[group_all.year_event.isna() == True, "observation"] = 0
# remplacer la date de la casse par 2013 la dernière date d'observation
group_all.loc[group_all.year_event.isna() == True, "duration"] = 2013 - group_all["year_pose"]

#group_all = group_all.dropna()

kmf = KaplanMeierFitter()

T = group_all["duration"]
E = group_all["observation"]

kmf.fit(T, event_observed=E, label = "collectivité 22")
print(kmf.survival_function_)

kmf.plot()








