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
    

# Path to data
PATH = "data/"

# Chargement dataset
df_all = pd.read_csv(PATH + 'master_df_all.csv')

df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

# récupérer les PVC pour TOUTES LES collectivités
df_MAT = df_all[df_all.MATERIAU == "PVC"]
df_MAT = df_MAT.drop(["ID" ], axis = 1)

fig, axes = plt.subplots(nrows = 5, ncols = 5, sharex = True,
                         sharey = True,figsize=(50, 35)
                        )
liste_col = df_MAT["collectivite"].unique()    
for col, ax in zip(liste_col, axes.flatten()):
    df_col = df_MAT[df_MAT.collectivite == col]
    categorical_km(df_col , col, ax = ax)
    
    ax.set_title(col,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
        
plt.tight_layout()



# récupérer les fonteductile pour TOUTES LES collectivités
df_font = df_all[df_all.MATERIAU == "FONTEDUCTILE"]
df_font = df_font.drop(["ID" ], axis = 1)

fig, axes = plt.subplots(nrows = 5, ncols = 5, sharex = True,
                         sharey = True,figsize=(50, 35)
                        )
liste_col = df_font["collectivite"].unique()    
for col, ax in zip(liste_col, axes.flatten()):
    df_col = df_font[df_font.collectivite == col]
    categorical_km(df_col , col, ax = ax)
    
    ax.set_title(col,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
        
plt.tight_layout()

# récupérer les plomb pour TOUTES LES collectivités
df_font = df_all[df_all.MATERIAU == "PLOMB"]
df_font = df_font.drop(["ID" ], axis = 1)

fig, axes = plt.subplots(nrows = 2, ncols = 2, sharex = True,
                         sharey = True,figsize=(20, 20)
                        )
liste_col = df_font["collectivite"].unique()    
for col, ax in zip(liste_col, axes.flatten()):
    df_col = df_font[df_font.collectivite == col]
    categorical_km(df_col , col, ax = ax)
    
    ax.set_title(col,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
        
plt.tight_layout()

# récupérer les plomb pour TOUTES LES collectivités
df_font = df_all[df_all.MATERIAU == "PEBD"]
df_font = df_font.drop(["ID" ], axis = 1)

fig, axes = plt.subplots(nrows = 2, ncols = 2, sharex = True,
                         sharey = True,figsize=(20, 20)
                        )
liste_col = df_font["collectivite"].unique()    
for col, ax in zip(liste_col, axes.flatten()):
    df_col = df_font[df_font.collectivite == col]
    categorical_km(df_col , col, ax = ax)
    
    ax.set_title(col,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
        
plt.tight_layout()