import pandas as pd
import numpy as np
from lifelines.utils import datetimes_to_durations
from lifelines.statistics import pairwise_logrank_test
import itertools
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt




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

#fonction pour les colonnes duration et event
def event_duration(df):
    df.loc[df.year_event.isna() == True, "duration"] = df.year_event.max() - df.year_event.min()
    df.loc[df.year_event.isna() == False, "duration"]= df['year_event'] - df.year_event.min()  

    T,E = datetimes_to_durations(df["DDP"], df["DDCC"], freq="Y")
    df["event"] = E

def calcul_Pvalue(colonne, df, liste_col):
    for m1, m2 in itertools.combinations(liste_col, 2):   
        df_2 = df.loc[(df[colonne] == m1) | (df[colonne] == m2)]
        event_duration(df_2)
        p = pairwise_logrank_test(df_2['duration'], df_2[colonne], df_2['event'])
        pValue = p.p_value[0]
        if round(pValue, 2)>0.05 :
            result_similar.append([ m1, m2, round(pValue, 2)])
        if round(pValue, 2)<=0.05 :
            result_dispar.append([m1, m2, round(pValue, 2)])
    return (result_dispar, result_similar)

def calcul_Pvalue_table(colonne, df, liste_col):
    table = np.zeros((len(liste_col), len(liste_col)))
    l1 = []
    l2 = []
    for m1, m2 in itertools.combinations(liste_col, 2):
        df_2 = df.loc[(df[colonne] == m1) | (df[colonne] == m2)]
        event_duration(df_2)
        p = pairwise_logrank_test(df_2['duration'], df_2[colonne], df_2['event'])
        pValue = p.p_value[0]
        table[liste_col.index(m1), liste_col.index(m2)] = round(pValue, 2)
        # l1.append(m1)
        # l2.append(m2)
        # l1_u = list(dict.fromkeys(l1))
        # l2_u = list(dict.fromkeys(l2))
    return table


#### test logrank pour chaque colonne de la collectivité 22 et pour chaque collectvité
col_list = [col for col in group_22.columns if group_22[col].dtype == object]
for col in col_list:
    result_similar = []
    result_dispar = []
    if col == "collectivite":
        liste_col = list(df_all["collectivite"].unique())
        #calcul_Pvalue(col, df_all, liste_col)
        table_collectivite = calcul_Pvalue_table(col, df_all, liste_col)
        #df_tab_pvalue_collect.to_csv('../results/Data_pvalue_collectivite.csv')
        np.savetxt('../results/tableau_collectivite.csv', table_collectivite, delimiter=',')
    else:
        liste_col = list(group_22[col].unique())
        calcul_Pvalue(col, group_22, liste_col)
            
    similar_data_collec = pd.DataFrame(result_similar, columns = ["Membre_1", "Membre_2", "p_value"])
    similar_data_collec.to_csv('../results/' + col+'_similar_pvalue.csv')
    
    dispart_data_collec = pd.DataFrame(result_dispar, columns = ["Membre_1", "Membre_2", "p_value"])
    dispart_data_collec.to_csv('../results/' + col+'_disparite_pvalue.csv')


## heatmap pour les p_values
liste_collect = df_all.collectivite.unique()

ax = sns.heatmap(table_collectivite,xticklabels=liste_collect, yticklabels=liste_collect,  annot = False,fmt='.2f', vmin=0, vmax=1, linewidth =0.05)

ax.set_xticklabels(ax.get_xticklabels(), fontsize = 8)
ax.set_yticklabels(ax.get_yticklabels(), fontsize = 8)

plt.title('Seaborn heatmap - pValue pour toutes les collectivités')

plt.show()




# #### logrank pour chaque colonne de la collectivité 22 et pour chaque collectvité
# liste_col = list(df_all["collectivite"].unique())
# for m1, m2 in itertools.combinations(liste_col, 2):   
#         df_2 = df_all.loc[(df_all["collectivite"] == m1) | (df_all["collectivite"] == m2)]
#         event_duration(df_2)
#         p = pairwise_logrank_test(df_2['duration'], df_2["collectivite"], df_2['event'])
# a = p.p_value[0]
# print(a)

