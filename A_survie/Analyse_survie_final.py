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
mpl.rcParams['lines.linewidth'] = 3

# Path to data
PATH = "data/"

# Chargement dataset
df_all = pd.read_csv(PATH + 'master_df_all.csv')

df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)


# récupérer toutes les données de la collectivité 22
df_collec = df_all[df_all.collectivite == "Collectivite_22"]
df_collec = df_collec.drop(["collectivite", "ID" ], axis = 1)

# récupérer les données pour les matériaus
## Set up subplot grid
plt.figure(1)

ax = plt.subplot(111)
df_mat1 = df_collec.loc[(df_collec.MATERIAU == "PVC") | 
                       (df_collec.MATERIAU == "PEHD")|
                       (df_collec.MATERIAU == "ACIER")|
                       (df_collec.MATERIAU == "FONTEDUCTILE")|
                       (df_collec.MATERIAU == "BONNA")]
df_mat2 = df_collec[df_collec.MATERIAU == "FONTEGRISE"]
liste_mat = np.array(["PVC, PEHD, ACIER, FONTEDUCTILE, BONNA", "FONTEGRISE"])

kmf = KaplanMeierFitter()
T1,E1 = datetimes_to_durations(df_mat1["DDP"], df_mat1["DDCC"], freq="Y")
kmf.fit(T1, event_observed=E1, label = liste_mat[0])    
kmf.survival_function_.plot(ax=ax, label = liste_mat[0])

T2,E2 = datetimes_to_durations(df_mat2["DDP"], df_mat2["DDCC"], freq="Y")
kmf.fit(T2, event_observed=E2, label = liste_mat[1])    
kmf.survival_function_.plot(ax=ax, label = liste_mat[1])

plt.title("Analyse de survie pour les différents matériaux")
plt.xlabel("Duration", fontsize= 10)
plt.ylabel("Survival probability", fontsize= 10)
plt.grid()



# récupérer les données pour un matage précis
plt.figure(2)
ax = plt.subplot(111)

df_mtg1 = df_collec.loc[(df_collec.MATAGE == "FONTEGRISE18001900")| 
                       (df_collec.MATAGE == "FONTEGRISE19401970")]
df_mtg2 = df_collec.loc[(df_collec.MATAGE == "FONTEGRISE19301940")|
                        (df_collec.MATAGE == "FONTEGRISE19001930")]

liste_mtg = ["FONTEGRISE18001900, FONTEGRISE19401970", "FONTEGRISE19301940, FONTEGRISE19001930"]
kmf = KaplanMeierFitter()
T1,E1 = datetimes_to_durations(df_mtg1["DDP"], df_mtg1["DDCC"], freq="Y")
kmf.fit(T1, event_observed=E1, label = liste_mtg[0])    
kmf.survival_function_.plot(ax=ax, label = liste_mtg[0])

T2,E2 = datetimes_to_durations(df_mtg2["DDP"], df_mtg2["DDCC"], freq="Y")
kmf.fit(T2, event_observed=E2, label = liste_mtg[1])    
kmf.survival_function_.plot(ax=ax, label = liste_mtg[1])

plt.title("Analyse de survie pour les différents matages")
plt.xlabel("Duration", fontsize= 10)
plt.ylabel("Survival probability", fontsize= 10)

plt.tight_layout()
plt.grid()


# récupérer les données pour un diametre précis
fig, axes = plt.subplots(nrows = 2, ncols = 3, sharex = True,
                         sharey = True,figsize=(40, 30)
                        )
#changer la configuration par défaut
#comportement 1
mpl.rcParams['lines.linewidth'] = 5

df_dmt1 = df_collec.loc[(df_collec.DIAMETRE == 5000) | 
                       (df_collec.DIAMETRE == 6000)|
                       (df_collec.DIAMETRE == 8000)|
                       (df_collec.DIAMETRE == 8100)]

df_dmt2 = df_collec.loc[(df_collec.DIAMETRE == 10000)|
                        (df_collec.DIAMETRE == 11000)|
                        (df_collec.DIAMETRE == 12500)|
                        (df_collec.DIAMETRE == 13500)|
                        (df_collec.DIAMETRE == 19000)]

df_dmt3 = df_collec.loc[(df_collec.DIAMETRE == 20000)|
                        (df_collec.DIAMETRE == 21600)|
                        (df_collec.DIAMETRE == 25000)]

df_dmt4 = df_collec.loc[(df_collec.DIAMETRE == 30000)|
                        (df_collec.DIAMETRE == 35000)]

df_dmt5 = df_collec.loc[(df_collec.DIAMETRE == 40000)|
                        (df_collec.DIAMETRE == 50000)|
                        (df_collec.DIAMETRE == 60000)|
                        (df_collec.DIAMETRE == 71000)|
                        (df_collec.DIAMETRE == 80000)]

df_dmt6 = df_collec.loc[(df_collec.DIAMETRE == 100000)|
                        (df_collec.DIAMETRE == 140000)]

list_dataframe = [df_dmt1, df_dmt2, df_dmt3, df_dmt4, df_dmt5, df_dmt6]
liste_dmt = ["5K, 6K, 8K et 8.1K", "10K, 11K, 12.5K, 13.5K et 19K", 
             "20K, 21.6K et 25K", "30K et 35K", "40K, 50K, 60K, 71K et 80K",
             "100K et 140K"]

for i, ax in zip(range(0,6), axes.flatten()):
    categorical_km(list_dataframe[i] , liste_dmt[i], ax = ax)
         
    ax.set_title(liste_dmt[i], pad=20,  fontsize=56)
    ax.set_xlabel('Duration', fontsize = 50)
    ax.set_ylabel('Survival probability', fontsize = 50)
    ax.grid()
    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=40)
        
plt.tight_layout()



# #comportement 2
# mpl.rcParams['lines.linewidth'] = 5
# plt.figure(3)

# ax = plt.subplot(111)
# mpl.rcParams['lines.linewidth'] = 3

# df_dmt7 = df_collec.loc[(df_collec.DIAMETRE == 2500) | 
#                         (df_collec.DIAMETRE == 3200)|
#                         (df_collec.DIAMETRE == 11400)|
#                         (df_collec.DIAMETRE == 14000)|
#                         (df_collec.DIAMETRE == 18000)|
#                         (df_collec.DIAMETRE == 21900)|
#                         (df_collec.DIAMETRE == 22500)]

# df_dmt8 = df_collec.loc[(df_collec.DIAMETRE == 155000)|
#                         (df_collec.DIAMETRE == 120000)|
#                         (df_collec.DIAMETRE == 4000)]

# liste_dmt2 = ["2.5K, 3.2K, 11.4K...", "4K, 10.8K, 155 et 120K"]

# kmf = KaplanMeierFitter()
# T1,E1 = datetimes_to_durations(df_dmt7["DDP"], df_dmt7["DDCC"], freq="Y")
# kmf.fit(T1, event_observed=E1, label = liste_dmt2[0])    
# kmf.survival_function_.plot(ax=ax, label = liste_dmt2[0])

# T2,E2 = datetimes_to_durations(df_dmt8["DDP"], df_dmt8["DDCC"], freq="Y")
# kmf.fit(T2, event_observed=E2, label = liste_dmt2[1])    
# kmf.survival_function_.plot(ax=ax, label = liste_dmt2[1])

# # T3,E3 = datetimes_to_durations(df_dmt1["DDP"], df_dmt1["DDCC"], freq="Y")
# # kmf.fit(T3, event_observed=E3, label = liste_dmt[0])    
# # kmf.survival_function_.plot(ax=ax, label = liste_dmt[0])

# plt.title("Analyse de survie pour un comportement différent selon le diamètre")
# plt.xlabel("Duration", fontsize= 10)
# plt.ylabel("Survival probability", fontsize= 10)
