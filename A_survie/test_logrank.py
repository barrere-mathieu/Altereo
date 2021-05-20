import pandas as pd
import numpy as np
import math 
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter
from lifelines.utils import datetimes_to_durations
from lifelines.statistics import multivariate_logrank_test
import itertools



from matplotlib import pyplot as plt
import plotly.graph_objs as go
from plotly import subplots
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import matplotlib as mpl
from matplotlib.offsetbox import AnchoredText

# Path to data
PATH = "../data/"

# Chargement dataset
df_all = pd.read_csv(PATH + 'master_df_all.csv')
df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)
df_all = df_all.drop(["ID"], axis = 1)

# préparation des données
df_all['DDP'] = pd.to_datetime(df_all['DDP'])
df_all['DDCC'] = pd.to_datetime(df_all['DDCC'])

df_all['year_pose'] = df_all['DDP'].apply(lambda x: x.year)
df_all['year_event'] = df_all['DDCC'].apply(lambda x: x.year)
df_all['diametre_range'] = pd.qcut(df_all['DIAMETRE'], q=6).astype(str)

group_22 = df_all[df_all.collectivite == "Collectivite_22"]

#fonction pour les colonnes diration et event
def event_duration(df):
    df.loc[df.year_event.isna() == True, "duration"] = df.year_event.max() - df.year_event.min()
    df.loc[df.year_event.isna() == False, "duration"]= df['year_event'] - df.year_event.min()  

    T,E = datetimes_to_durations(df["DDP"], df["DDCC"], freq="Y")
    df["event"] = E

def calcul_Pvalue(colonne, df, liste_col):
    for m1, m2 in itertools.combinations(liste_col, 2):   
        df_2 = df.loc[(df[colonne] == m1) | (df[colonne] == m2)]
        event_duration(df_2)
        p = multivariate_logrank_test(df_2['duration'], df_2[colonne], df_2['event'])
        if round(p.p_value, 2)>0.05 :
            result_similar.append([ m1, m2, round(p.p_value, 2)])
        if round(p.p_value, 2)<=0.05 :
            result_dispar.append([m1, m2, round(p.p_value, 2)])
    return (result_dispar, result_similar)

def calcul_Pvalue_table(colonne, df, liste_col):
    table = np.zeros((len(liste_col), len(liste_col)))
    for m1, m2 in itertools.combinations(liste_col, 2):
        df_2 = df.loc[(df[colonne] == m1) | (df[colonne] == m2)]
        event_duration(df_2)
        p = multivariate_logrank_test(df_2['duration'], df_2[colonne], df_2['event'])
        table[liste_col.index(m1), liste_col.index(m2)] = round(p.p_value, 2)
    return table

#### test logrank pour chaque colonne de la collectivité 22 et pour chaque collectvité
col_list = [col for col in group_22.columns if group_22[col].dtype == object]
for col in col_list:
    result_similar = []
    result_dispar = []
    if col == "collectivite":
        liste_col = list(df_all["collectivite"].unique())
        calcul_Pvalue(col, df_all, liste_col)
        table_collectivite = calcul_Pvalue_table(col, df_all, liste_col)
        np.savetxt('../results/tableau_collectivite.csv', table_collectivite, delimiter=',')
    else:
        liste_col = list(group_22[col].unique())
        calcul_Pvalue(col, group_22, liste_col)
            
    similar_data_collec = pd.DataFrame(result_similar, columns = ["Membre_1", "Membre_2", "p_value"])
    similar_data_collec.to_csv('../results/' + col+'_similar_pvalue.csv')
    
    dispart_data_collec = pd.DataFrame(result_dispar, columns = ["Membre_1", "Membre_2", "p_value"])
    dispart_data_collec.to_csv('../results/' + col+'_disparite_pvalue.csv')


# #### test logrank pour chaque collectvité

# liste_col = list(df_all["collectivite"].unique())
# result_similar = []
# result_dispar = []
# calcul_Pvalue("collectivite", df_all, liste_col)     
# similar_data_collec = pd.DataFrame(result_similar, columns = ["Membre_1", "Membre_2", "p_value"])
# similar_data_collec.to_csv(PATH + 'collet_similar_pvalue.csv')

# dispart_data_collec = pd.DataFrame(result_dispar, columns = ["Membre_1", "Membre_2", "p_value"])
# dispart_data_collec.to_csv(PATH + 'collect_disparite_pvalue.csv')



# col_list = [col for col in group_22.columns if group_22[col].dtype == object]
# del col_list[2]
# for col in col_list:
#     #Materiaux
#     liste_col = list(group_22[col].unique()) # liste des materiaux
#     result_similar = []
#     result_dispar = []
#     for m1, m2 in itertools.combinations(liste_col, 2):   
#         df_2 = group_22.loc[(group_22[col] == m1) | (group_22[col] == m2)]
#         event_duration(df_2)
#         p = multivariate_logrank_test(df_2['duration'], df_2[col], df_2['event'])
#         if round(p.p_value, 2)>0.05 :
#             result_dispar.append([ m1, m2, round(p.p_value, 2)])
#         if round(p.p_value, 2)<=0.05 :
#             result_similar.append([m1, m2, round(p.p_value, 2)])
            
#     similar_data_collec = pd.DataFrame(result_similar, columns = ["Membre_1", "Membre_2", "p_value"])
#     similar_data_collec.to_csv(PATH + col+'_similar_pvalue.csv')
    
#     dispart_data_collec = pd.DataFrame(result_dispar, columns = ["Membre_1", "Membre_2", "p_value"])
#     dispart_data_collec.to_csv(PATH + col+'_disparite_pvalue.csv')
#################################################


# #fonction pour appliquer Kapplan_meier
# def categorical_km(feature, t='duration', event='event', df=group_22, ax=None):
#     df.loc[df.year_event.isna() == True, "duration"] = df.year_event.max() - df.year_event.min()
#     df.loc[df.year_event.isna() == False, "duration"]= df['year_event'] - df.year_event.min()  

#     T,E = datetimes_to_durations(df["DDP"], df["DDCC"], freq="Y")
#     df["event"] = E
#     for cat in df[feature].unique():
#         idx = df[feature] == cat
#         kmf = KaplanMeierFitter()

#         kmf.fit(df[idx][t], event_observed=df[idx][event], label=cat)    
#         kmf.plot(ax=ax, label=cat, ci_show=True, legend=False)
    
#fonction pour les colonnes diration et event


# col_list = ["MATERIAU"]

# fig, axes = plt.subplots(nrows = 1, ncols = 1, 
#                          sharex = True, sharey = True,
#                          figsize=(15, 10)
#                         )

# for cat, ax in zip(col_list, axes.flatten()):
#     categorical_km(feature=cat, t='duration', event='event', df = group_22, ax=ax)
#     ax.legend(loc='lower left', prop=dict(size=14))
#     ax.set_title(cat, pad=20, fontsize=56)
#     p = multivariate_logrank_test(group_22['duration'], group_22[cat], group_22['event'])
#     ax.add_artist(AnchoredText(round(p.p_value, 2), frameon=False, 
#                                 loc='lower right', prop=dict(size=46)))
#     ax.set_xlabel('Duration', fontsize = 40)
#     ax.set_ylabel('Survival probability', fontsize = 40)
#     ax.grid()
#     ax.tick_params(axis='x', labelsize=30)
#     ax.tick_params(axis='y', labelsize=30)
    
# fig.tight_layout() 


# col_list = ["diametre_range"]
# fig, axes = plt.subplots(nrows = 1, ncols = 1, 
#                          sharex = True, sharey = True,
#                          figsize=(15, 10)
#                         )
# for cat in col_list:
#     categorical_km(feature=cat, t='duration', event='event', df = df_all, ax=axes)
#     axes.legend(loc='lower left', prop=dict(size=14))
#     axes.set_title(cat, pad=20, fontsize=56)
#     p = multivariate_logrank_test(df_all['duration'], df_all[cat], df_all['event'])
#     axes.add_artist(AnchoredText(round(p.p_value, 2), frameon=False, 
#                                loc='lower right', prop=dict(size=46)))
#     axes.set_xlabel('Duration', fontsize = 40)
#     axes.set_ylabel('Survival probability', fontsize = 40)
#     axes.grid()
#     axes.tick_params(axis='x', labelsize=30)
#     axes.tick_params(axis='y', labelsize=30)
    
# fig.tight_layout() 
