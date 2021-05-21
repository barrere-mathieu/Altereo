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



#fonction pour les colonnes duration et event
def event_duration(df):
    df.loc[df.year_event.isna() == True, "duration"] = df.year_event.max() - df.year_event.min()
    df.loc[df.year_event.isna() == False, "duration"]= df['year_event'] - df.year_event.min()  

    T,E = datetimes_to_durations(df["DDP"], df["DDCC"], freq="Y")
    df["event"] = E
    

def calcul_Pvalue_table(colonne, df, liste_col):
    table = np.zeros((len(liste_col), len(liste_col)))
    for m1, m2 in itertools.combinations(liste_col, 2):
        df_2 = df.loc[(df[colonne] == m1) | (df[colonne] == m2)]
        event_duration(df_2)
        p = pairwise_logrank_test(df_2['duration'], df_2[colonne], df_2['event'])
        pValue = p.p_value[0]
        table[liste_col.index(m1), liste_col.index(m2)] = round(pValue, 2)
    return table

#### test logrank pour chaque colonne de data_all
col_list = [col for col in df_all.columns if df_all[col].dtype == object]
for col in col_list:
    liste_col = list(df_all[col].unique())
    table_collectivite = calcul_Pvalue_table(col, df_all, liste_col)
    np.savetxt('../results/tableau_'+col+'.csv', table_collectivite, delimiter=',')
    
    ax = sns.heatmap(table_collectivite, xticklabels=liste_col, yticklabels=liste_col, vmin=0, vmax=1, linewidth =0.05)
    plt.title('Seaborn heatmap - pValue pour:'+ col) 
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 8)

