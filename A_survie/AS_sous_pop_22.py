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
    
    
#fonction pour appliquer Kapplan_meier
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
    
    
#changer la configuration par défaut
mpl.rcParams['lines.linewidth'] = 7

# Path to data
PATH = "../data/"

# Chargement dataset
df_all = pd.read_csv(PATH + 'master_df_all.csv')

df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)
df_all = df_all.drop(["ID"], axis = 1)


# récupérer toutes les données de la collectivité 22
year_event_pose(df_all)
df_22 = df_all[df_all.collectivite == "Collectivite_22"]

# récupérer les données pour un matérieu précis
## Set up subplot grid
fig, axes = plt.subplots(nrows = 2, ncols = 3, sharex = False,
                         sharey = False,figsize=(25, 20)
                        )

liste_mat = df_22["MATERIAU"].unique()
for mat, ax in zip(liste_mat, axes.flatten()):
    df_mat = df_22[df_22.MATERIAU == mat]
    categorical_km(df_22, df_mat , mat, ax = ax)
    
    ax.set_title(mat,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    
fig.tight_layout()


# récupérer les données pour un matage précis
fig, axes = plt.subplots(nrows = 5, ncols = 5, sharex = False,
                         sharey = False,figsize=(50, 35)
                        )
liste_mtg = df_22["MATAGE"].unique()    
for mtg, ax in zip(liste_mtg, axes.flatten()):
    df_mtg = df_22[df_22.MATAGE == mtg]
    categorical_km(df_22, df_mtg , mtg, ax = ax)
    
    ax.set_title(mtg,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
        
fig.tight_layout()

# récupérer les données pour un rang de diametre précis
fig, axes = plt.subplots(nrows = 2, ncols = 3, sharex = False,
                         sharey = False,figsize=(25, 15)
                        )
liste_dmt = sorted(df_22["diametre_range"].unique())

for dmt, ax in zip(liste_dmt, axes.flatten()):
    df_dmt = df_22[df_22.diametre_range == dmt]
    categorical_km(df_22, df_dmt , dmt, ax = ax)
    
    ax.set_title(dmt,pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 40)
    ax.set_ylabel('Survival probability', fontsize = 40)
    ax.grid()
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
        
fig.tight_layout()




# ########## test
# df_mdt = df_collec[df_collec.MATERIAU == "PEHD"]

# # pas de casse dans tte la liste
# if df_mdt.year_event.isnull().values.all() == True:
#     print("hello 11111111111111111111111111")
#     df_mdt.loc[df_mdt.year_event.isna() == True, "duration"] = df_collec.year_event.max()- df_collec.year_event.min()
# #une seule casse
# elif df_mdt.year_event.max() == df_mdt.year_event.min() and df_mdt.year_event.isnull().values.all() == False:
#     print("hello")
#     df_mdt.loc[df_mdt.year_event.isna() == True, "duration"] = df_collec.year_event.max()- df_collec.year_event.min()
#     df_mdt.loc[df_mdt.year_event.isna() == False, "duration"]= df_mdt['year_event']- df_collec.year_event.min() 
# else:
#     print("Olfaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
#     df_mdt.loc[df_mdt.year_event.isna() == True, "duration"] = df_collec.year_event.max() - df_collec.year_event.min()
#     df_mdt.loc[df_mdt.year_event.isna() == False, "duration"]= df_mdt['year_event'] - df_collec.year_event.min()  

# T,E = datetimes_to_durations(df_mdt["DDP"], df_mdt["DDCC"], freq="Y")
# kmf = KaplanMeierFitter()

# kmf.fit(df_mdt.duration, event_observed=E, label = 16200)    
# kmf.plot(label = 16200, legend=False)



# # récupérer les données pour un diametre précis
# fig, axes = plt.subplots(nrows = 10, ncols = 5, sharex = True,
#                          sharey = True,figsize=(50, 80)
#                         )
# liste_dmt = sorted(df_collec["DIAMETRE"].unique())
# for dmt, ax in zip(liste_dmt, axes.flatten()):
#     df_dmt = df_collec[df_collec.DIAMETRE == dmt]
#     categorical_km(df_collec, df_dmt , dmt, ax = ax)
    
#     ax.set_title(dmt,pad=20,  fontsize=56)
#     ax.set_xlabel('Duration', fontsize = 40)
#     ax.set_ylabel('Survival probability', fontsize = 40)
#     ax.grid()
#     ax.tick_params(axis='x', labelsize=24)
#     ax.tick_params(axis='y', labelsize=24)
        
# plt.tight_layout()


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