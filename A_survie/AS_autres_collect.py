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

# fonction pour préparer les données
def prep_donnees(df):
    df['DDP'] = pd.to_datetime(df_all['DDP'])
    df['DDCC'] = pd.to_datetime(df_all['DDCC'])
    
    df['year_pose'] = df['DDP'].apply(lambda x: x.year)
    df['year_event'] = df['DDCC'].apply(lambda x: x.year)
    df = df.drop(["ID"], axis = 1)


#fonction pour appliquer Kapplan_meier
def categorical_km(df, df_cat ,cat, ax = None):
    # pas de casse dans tt la liste
    if df_cat.year_event.isnull().values.all() == True:
        df_cat.loc[df_cat.year_event.isna() == True, "duration"] = df.year_event.max()- df.year_event.min()
    else:
        #df_cat.year_event.isnull().values.all() == False:
        df_cat.loc[df_cat.year_event.isna() == True, "duration"] = df.year_event.max() - df.year_event.min()
        df_cat.loc[df_cat.year_event.isna() == False, "duration"]= df_cat['year_event'] - df.year_event.min()  
    
    T,E = datetimes_to_durations(df_cat["DDP"], df_cat["DDCC"], freq="Y")
    kmf = KaplanMeierFitter()
    
    kmf.fit(df_cat.duration, event_observed=E, label = cat)    
    kmf.plot(ax=ax, label = cat, legend=False)
    
#changer la configuration par défaut
mpl.rcParams['lines.linewidth'] = 7

# Path to data
PATH = "../data/"

# Chargement dataset
df_all = pd.read_csv(PATH + 'master_df_all.csv')

df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

# préparation des données
prep_donnees(df_all)


# récupérer le PVC  et le fonteductile de chaque collectivité
liste_mat = ["FONTEDUCTILE", "PVC"]
for mat in liste_mat:
    fig, axes = plt.subplots(nrows = 5, ncols = 5, sharex = True,
                             sharey = True,figsize=(50, 35)
                            )
    #liste_col = df_all["collectivite"].unique()
    liste_col = df_all[df_all.MATERIAU == mat].collectivite.unique()    
    for col, ax in zip(liste_col, axes.flatten()):
        df_col = df_all[df_all.collectivite == col ]
        df_mat = df_col[df_all.MATERIAU == mat]
        categorical_km(df_col, df_mat , col, ax = ax)
        
        ax.set_title(col,pad=20,  fontsize=56)
        ax.set_xlabel('Duration', fontsize = 40)
        ax.set_ylabel('Survival probability', fontsize = 40)
        ax.grid()
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        

# récupérer les plomb pour chaque collectivité
liste_mat = ["PLOMB", "PEBD"]
for mat in liste_mat:
    fig, axes = plt.subplots(nrows = 2, ncols = 2, sharex = True,
                         sharey = True,figsize=(20, 20)
                        )       
    liste_col = df_all[df_all.MATERIAU == mat].collectivite.unique()
    for col, ax in zip(liste_col, axes.flatten()):
        df_col = df_all[df_all.collectivite == col ]
        df_mat = df_col[df_all.MATERIAU == mat]
        categorical_km(df_col, df_mat , col, ax = ax)
        
        ax.set_title(col,pad=20,  fontsize=56)
        ax.set_xlabel('Duration', fontsize = 40)
        ax.set_ylabel('Survival probability', fontsize = 40)
        ax.grid()
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
    

# ########## test
# df_collec = df_all[df_all.collectivite == "Collectivite_12" ]
# df_mdt = df_all.loc[(df_all.MATERIAU == "PLOMB") & (df_all.collectivite == "Collectivite_12")]

# # pas de casse dans tte la liste
# if df_mdt.year_event.isnull().values.all() == True:
#     print("hello 11111111111111111111111111")
#     df_mdt.loc[df_mdt.year_event.isna() == True, "duration"] = df_collec.year_event.max()- df_collec.year_event.min()
# else:
#     print("Olfaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
#     df_mdt.loc[df_mdt.year_event.isna() == True, "duration"] = df_collec.year_event.max() - df_collec.year_event.min()
#     df_mdt.loc[df_mdt.year_event.isna() == False, "duration"]= df_mdt['year_event'] - df_collec.year_event.min()  

# T,E = datetimes_to_durations(df_mdt["DDP"], df_mdt["DDCC"], freq="Y")
# kmf = KaplanMeierFitter()

# kmf.fit(df_mdt.duration, event_observed=E, label = 16200)    
# kmf.plot(label = 16200, legend=False)
