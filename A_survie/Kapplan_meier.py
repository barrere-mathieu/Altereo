import pandas as pd
import numpy as np
import math 
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter
from lifelines.utils import datetimes_to_durations

from matplotlib import pyplot as plt
import plotly.graph_objs as go
from plotly import subplots
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import matplotlib as mpl


#fonction pour appliquer Kapplan_meier
def categorical_km(df,cat, ax = None):
    df.loc[df.year_event.isna() == True, "duration"] = df.year_event.max() - df.year_event.min()
    df.loc[df.year_event.isna() == False, "duration"]= df['year_event'] - df.year_event.min()  

    T,E = datetimes_to_durations(df["year_pose"], df["year_event"], freq="Y")
    kmf = KaplanMeierFitter()

    kmf.fit(df.duration, event_observed=E, label = cat)    
    kmf.plot(ax=ax, label = cat, legend=False)

# Path to data
PATH = "../data/"

# Chargement dataset
df_all = pd.read_csv(PATH + 'master_df_all.csv')
df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)


# préparation des données
df_all['DDP'] = pd.to_datetime(df_all['DDP'])
df_all['DDCC'] = pd.to_datetime(df_all['DDCC'])

df_all['year_pose'] = df_all['DDP'].apply(lambda x: x.year)
df_all['year_event'] = df_all['DDCC'].apply(lambda x: x.year)
df_all = df_all.drop(["DDP", "DDCC", "ID" ], axis = 1)

# Analyse de survie independemment des collectivités
#changer la configuration par défaut
mpl.rcParams['lines.linewidth'] = 2

# remplacer la date de la casse par 2013 (la dernière date d'observation)
df_all.loc[df_all.year_event.isna() == True, "duration"] = df_all.year_event.max() - df_all.year_event.min()
df_all.loc[df_all.year_event.isna() == False, "duration"]= df_all['year_event'] - df_all.year_event.min()  

T,E = datetimes_to_durations(df_all["year_pose"], df_all["year_event"], freq="Y")

kmf = KaplanMeierFitter()

kmf.fit(df_all.duration, event_observed=E, label = "Toutes les collectivités")
kmf.plot()


# récupérer toutes les données de la collectivité 22
group_22 = df_all[df_all.collectivite == "Collectivite_22"]

group_22.loc[group_22.year_event.isna() == True, "duration"] = group_22.year_event.max() - group_22.year_event.min()
group_22.loc[group_22.year_event.isna() == False, "duration"]= group_22['year_event'] - group_22.year_event.min()  

T2,E2 = datetimes_to_durations(group_22["year_pose"], group_22["year_event"], freq="Y")

kmf2 = KaplanMeierFitter()

kmf2.fit(group_22.duration, event_observed=E2, label = "Collectivité 22")
#print(kmf2.survival_function_)

kmf2.plot()
plt.grid()

# récupérer les données pour chaque collectivité
#changer la configuration par défaut
mpl.rcParams['lines.linewidth'] = 5

fig, axes = plt.subplots(nrows = 5, ncols = 5, sharex = True,
                          sharey = True,figsize=(50, 40)
                        )
liste_col = df_all["collectivite"].unique()
for col, ax in zip(liste_col, axes.flatten()):
    df_col = df_all[df_all.collectivite == col]
    categorical_km(df_col , col, ax = ax)
    
    ax.set_title(col,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
        
plt.tight_layout()








# # Analyse de survie independemment des collectivités
# df_all["duration"] = df_all['year_event'] - df_all["year_pose"]
# df_all["observation"] = 1  # 1 pour tuyaux pas cassé
# # remplacer l'état des tuyaux cassé de 1 --> 0
# df_all.loc[df_all.year_event.isna() == True, "observation"] = 0
# # remplacer la date de la casse par 2013 la dernière date d'observation
# df_all.loc[df_all.year_event.isna() == True, "duration"] = 2013 - df_all["year_pose"]

# kmf = KaplanMeierFitter()

# T = df_all["duration"]
# E = df_all["observation"]

# kmf.fit(T, event_observed=E, label = "Toutes les collectivités")
# print(kmf.survival_function_)

# kmf.survival_function_.plot()

# # récupérer toutes les données de la collectivité 22
# group_all = df_all[df_all.collectivite == "Collectivite_22"]

# group_all["duration"] = group_all['year_event'] - group_all["year_pose"]
# a= group_all['year_event'].values
# group_all["observation"] = 1  # 0 pour tuyaux pas cassé
# # remplacer l'état des tuyaux cassé de 1 --> 0
# group_all.loc[group_all.year_event.isna() == True, "observation"] = 0
# # remplacer la date de la casse par 2013 la dernière date d'observation
# group_all.loc[group_all.year_event.isna() == True, "duration"] = 2013 - group_all["year_pose"]

# kmf = KaplanMeierFitter()

# T = group_all["duration"]
# E = group_all["observation"]

# kmf.fit(T, event_observed=E, label = "Collectivité 22")
# print(kmf.survival_function_)

# kmf.plot()




# # cumulative density: prob qu'un tuyaux se casse dans les deux cas précédents
# plt.figure(2)

# kmf.plot_cumulative_density()
# kmf2.plot_cumulative_density()
# plt.grid()







