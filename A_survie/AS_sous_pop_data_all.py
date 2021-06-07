import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import datetimes_to_durations

from matplotlib import pyplot as plt
import plotly.io as pio
pio.renderers.default = "browser"
import matplotlib as mpl

# Ajout des colonnes "year_pose" et "year_event"
def year_event_pose(df):
    df['DDP'] = pd.to_datetime(df['DDP'])
    df['DDCC'] = pd.to_datetime(df['DDCC'])
    
    df['year_pose'] = df['DDP'].apply(lambda x: x.year)
    df['year_event'] = df['DDCC'].apply(lambda x: x.year)
    df['diametre_range'] = pd.qcut(df['DIAMETRE'], q=6).astype(str)
    
  
#fonction pour appliquer Kapplan_meier (fenètre d'observation de data_all )
def categorical_km(df, df_cat ,cat, ax = None):
    # pas de casse dans tt la liste
    if df_cat.year_event.isnull().values.all() == True:
        df_cat.loc[df_cat.year_event.isna() == True, "duration"] = df.year_event.max()- df.year_event.min()
    else:
        df_cat.loc[df_cat.year_event.isna() == True, "duration"] = df.year_event.max() - df.year_event.min()
        df_cat.loc[df_cat.year_event.isna() == False, "duration"]= df_cat['year_event'] - df.year_event.min()  
    
    T,E = datetimes_to_durations(df_cat["DDP"], df_cat["DDCC"], freq="Y")
    kmf = KaplanMeierFitter()
    
    kmf.fit(df_cat.duration, event_observed=E, label = cat)    
    kmf.plot(ax=ax, label = cat, legend=False)

#fonction pour appliquer Kapplan_meier (fenètre d'observation de la sous population)
def categorical_km1(df,cat, ax = None):
    if df.year_event.isnull().values.all() == True:
        df.loc[df.year_event.isna() == True, "duration"] = df.year_event.max()- df.year_event.min()
    else:
        df.loc[df.year_event.isna() == True, "duration"] = df.year_event.max() - df.year_event.min()
        df.loc[df.year_event.isna() == False, "duration"]= df['year_event'] - df.year_event.min()  
    
    T,E = datetimes_to_durations(df["DDP"], df["DDCC"], freq="Y")
    df["event"] = E
    kmf = KaplanMeierFitter()

    kmf.fit(df.duration, event_observed=E, label = cat)    
    kmf.plot(ax=ax, label = cat, legend=False)
    
#changer la configuration par défaut
mpl.rcParams['lines.linewidth'] = 7

# Path to data
PATH = "../data/"

# Chargement dataset
df_all = pd.read_csv(PATH + 'master_df_all.csv')

df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)
df_all = df_all.drop(["ID"], axis = 1)


# récupérer toutes les données des tuyaux cassés
year_event_pose(df_all)

# Analyse de survie pour chaque matérieu
## Set up subplot grid
fig, axes = plt.subplots(nrows = 2, ncols = 4, sharex = False,
                         sharey = False,figsize=(25, 20)
                        )

liste_mat = df_all["MATERIAU"].unique()
for mat, ax in zip(liste_mat, axes.flatten()):
    df_mat = df_all[df_all.MATERIAU == mat]
    categorical_km(df_all, df_mat , mat, ax = ax)
    #categorical_km1(df_mat,mat, ax = ax)
    
    ax.set_title(mat,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    
fig.tight_layout()


# Analyse de survie pour chaque matage
fig, axes = plt.subplots(nrows = 5, ncols = 5, sharex = False,
                         sharey = False,figsize=(50, 35)
                        )
liste_mtg = df_all["MATAGE"].unique()    
for mtg, ax in zip(liste_mtg, axes.flatten()):
    df_mtg = df_all[df_all.MATAGE == mtg]
    categorical_km(df_all, df_mtg , mtg, ax = ax) #fenètre d'observation de data_all
    #categorical_km1(df_mtg,mtg, ax = ax)   #fenètre d'observation de df_mtg
    
    ax.set_title(mtg,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
        
fig.tight_layout()


# Analyse de survie pour chaque tranche de diametre 
fig, axes = plt.subplots(nrows = 2, ncols = 3, sharex = False,
                         sharey = False,figsize=(25, 15)
                        )
liste_dmt = sorted(df_all["diametre_range"].unique())

for dmt, ax in zip(liste_dmt, axes.flatten()):
    df_dmt = df_all[df_all.diametre_range == dmt]
    categorical_km(df_all, df_dmt , dmt, ax = ax)
    #categorical_km1(df_dmt,dmt, ax = ax)
    
    ax.set_title(dmt,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
        
fig.tight_layout()


# Analyse de survie independemment des collectivités
#changer la configuration par défaut
mpl.rcParams['lines.linewidth'] = 2

fig, axes = plt.subplots(nrows = 5, ncols = 5, sharex = False,
                          sharey = False,figsize=(50, 40)
                        )
liste_col = df_all["collectivite"].unique()
for col, ax in zip(liste_col, axes.flatten()):
    df_col = df_all[df_all.collectivite == col]
    categorical_km1(df_col , col, ax = ax)
    ax.set_title(col,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
        
plt.tight_layout()


# ########## test
# df_test = df_all[df_all.MATERIAU == "PLOMB"]
# df_test = df_all[df_all.MATAGE == "ACIER18451920"]

# a = df_test.year_event.max()
# b = df_test.year_event.min()

# if df_test.year_event.isnull().values.all() == True:
#         df_test.loc[df_test.year_event.isna() == True, "duration"] = df_test.year_event.max()- df_test.year_event.min()
# else:
#     df_test.loc[df_test.year_event.isna() == True, "duration"] = df_test.year_event.max() - df_test.year_event.min()
#     df_test.loc[df_test.year_event.isna() == False, "duration"]= df_test['year_event'] - df_test.year_event.min()  

# T,E = datetimes_to_durations(df_test["DDP"], df_test["DDCC"], freq="Y")
# df_test["event"] = E
# kmf = KaplanMeierFitter()

# kmf.fit(df_test.duration, event_observed=E)    
# kmf.plot(label = "PLOMB", legend=False)