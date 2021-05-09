import pandas as pd
import numpy as np
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter
from lifelines.utils import datetimes_to_durations


from matplotlib import pyplot as plt
from plotly import subplots
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import matplotlib as mpl

#fonction pour appliquer Kapplan_meier
def categorical_km(df,cat, ax = None):
    T,E = datetimes_to_durations(df["DDP"], df["DDCC"], freq="Y")
    kmf = KaplanMeierFitter()

    kmf.fit(T, event_observed=E, label = cat)    
    kmf.survival_function_.plot(ax=ax, label = cat, legend=False)

#changer la configuration par défaut
mpl.rcParams['lines.linewidth'] = 7

# Path to data
PATH = "data/"

# Chargement dataset
df_all = pd.read_csv(PATH + 'master_df_all.csv')

df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)


# récupérer toutes les données de la collectivité 22
df_collec = df_all[df_all.collectivite == "Collectivite_22"]
df_collec = df_collec.drop(["collectivite", "ID" ], axis = 1)

# récupérer les données pour un matérieu précis
## Set up subplot grid
fig, axes = plt.subplots(nrows = 2, ncols = 3, sharex = True,
                         sharey = True,figsize=(25, 20)
                        )

liste_mat = df_collec["MATERIAU"].unique()
for mat, ax in zip(liste_mat, axes.flatten()):
    df_mat = df_collec[df_collec.MATERIAU == mat]
    categorical_km(df_mat , mat, ax = ax)
    
    ax.set_title(mat,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)

    
plt.tight_layout()


# récupérer les données pour un matage précis
fig, axes = plt.subplots(nrows = 5, ncols = 5, sharex = True,
                         sharey = True,figsize=(50, 35)
                        )
liste_mtg = df_collec["MATAGE"].unique()    
for mtg, ax in zip(liste_mtg, axes.flatten()):
    df_mtg = df_collec[df_collec.MATAGE == mtg]
    categorical_km(df_mtg , mtg, ax = ax)
    
    ax.set_title(mtg,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
        
plt.tight_layout()


# récupérer les données pour un diametre précis
fig, axes = plt.subplots(nrows = 10, ncols = 5, sharex = True,
                         sharey = True,figsize=(50, 80)
                        )
liste_dmt = sorted(df_collec["DIAMETRE"].unique())
for dmt, ax in zip(liste_dmt, axes.flatten()):
    df_dmt = df_collec[df_collec.DIAMETRE == dmt]
    categorical_km(df_dmt , dmt, ax = ax)
    
    ax.set_title(dmt,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
        
plt.tight_layout()

################### test 2
# liste_mat = df_collec["MATERIAU"].unique()
# df_mat = df_collec[df_collec.MATERIAU == "FONTEDUCTILE"]
# ax = plt.subplot(111)
# kmf = KaplanMeierFitter()
# T,E = datetimes_to_durations(df_mat["DDP"], df_mat["DDCC"], freq="D")

# kmf.fit(T, event_observed=E, label = "FONTEDUCTILE")

# kmf.survival_function_.plot(ax = ax)

# df_mat = df_collec[df_collec.MATERIAU == "ACIER"]
    
# kmf = KaplanMeierFitter()
# T,E = datetimes_to_durations(df_mat["DDP"], df_mat["DDCC"], freq="D")

# kmf.fit(T, event_observed=E, label = "ACIER")

# kmf.survival_function_.plot(ax = ax)

# # récupérer les données pour un matérieu précis
# liste_mat = df_collec["MATERIAU"].unique()
# for i, mat in enumerate(liste_mat):
#     ax = plt.subplot(2,3,i+1)
#     df_mat = df_collec[df_collec.MATERIAU == mat]
    
#     T,E = datetimes_to_durations(df_mat["DDP"], df_mat["DDCC"], freq="Y")
#     kmf = KaplanMeierFitter()
    
#     kmf.fit(T, event_observed=E, label = mat)    
#     kmf.plot_survival_function(ax=ax, legend=False)
    
#     plt.title(mat)
    
#     if i==0:
#         plt.ylabel('Frac. in power after $n$ years')
        
# plt.tight_layout()