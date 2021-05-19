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
PATH = "data/"

# Chargement dataset
df_all = pd.read_csv(PATH + 'master_df_all.csv')
df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

# préparation des données
df_all['DDP'] = pd.to_datetime(df_all['DDP'])
df_all['DDCC'] = pd.to_datetime(df_all['DDCC'])

df_all['year_pose'] = df_all['DDP'].apply(lambda x: x.year)
df_all['year_event'] = df_all['DDCC'].apply(lambda x: x.year)
df_all['diametre_range'] = pd.qcut(df_all['DIAMETRE'], q=6).astype(str)


#fonction pour les colonnes diration et event
def event_duration(df):
    df.loc[df.year_event.isna() == True, "duration"] = df.year_event.max() - df.year_event.min()
    df.loc[df.year_event.isna() == False, "duration"]= df['year_event'] - df.year_event.min()  

    T,E = datetimes_to_durations(df["DDP"], df["DDCC"], freq="Y")
    df["event"] = E

collec = list(df_all.collectivite.unique())

result_similar = []
result_dispar = []
for c1, c2 in itertools.combinations(collec, 2):   
    df_2 = df_all.loc[(df_all.collectivite == c1) | (df_all.collectivite == c2)]
    event_duration(df_2)
    p = multivariate_logrank_test(df_2['duration'], df_2["collectivite"], df_2['event'])
    if round(p.p_value, 2)>0.05 :
        result_similar.append([ c1, c2, round(p.p_value, 2)])
    if round(p.p_value, 2)<=0.05 :
        result_dispar.append([c1, c2, round(p.p_value, 2)])
        
similar_data_collec = pd.DataFrame(result_similar, columns = ["Collectivite_1", "Collectivite_2", "p_value"])
similar_data_collec.to_csv(PATH + 'similar_data_collect.csv')

dispart_data_collec = pd.DataFrame(result_dispar, columns = ["Collectivite_1", "Collectivite_2", "p_value"])
dispart_data_collec.to_csv(PATH + 'disparite_data_collect.csv')
        
        

# for e in result:
#     print("Catégorie: {} - Collectivités {} vs. {} : LOGRANK p value = {}".format(e[0], e[1],e[2],e[3] ))
    


# #fonction pour appliquer Kapplan_meier
# def categorical_km(feature, t='duration', event='event', df=df_all, ax=None):
#     # df.loc[df.year_event.isna() == True, "duration"] = df.year_event.max() - df.year_event.min()
#     # df.loc[df.year_event.isna() == False, "duration"]= df['year_event'] - df.year_event.min()  

#     # T,E = datetimes_to_durations(df["DDP"], df["DDCC"], freq="Y")
#     # df["event"] = E
#     for cat in df[feature].unique():
#         idx = df[feature] == cat
#         kmf = KaplanMeierFitter()

#         kmf.fit(df[idx][t], event_observed=df[idx][event], label=cat)    
#         kmf.plot(ax=ax, label=cat, ci_show=True, legend=False)
# fig, axes = plt.subplots(nrows = 1, ncols = 1, 
#                          sharex = True, sharey = True,
#                          figsize=(15, 10)
#                         )

# for cat in col_list:
#     categorical_km(feature=cat, t='duration', event='event', df = df_all, ax=axes)
#     axes.legend(loc='lower left', prop=dict(size=14))
#     axes.set_title(cat, pad=20, fontsize=56)
#     p = multivariate_logrank_test(df_all['duration'], df_all[cat], df_all['event'])
#     axes.add_artist(AnchoredText(p.p_value, frameon=False, 
#                                loc='upper right', prop=dict(size=46)))
#     axes.set_xlabel('Duration', fontsize = 40)
#     axes.set_ylabel('Survival probability', fontsize = 40)
#     axes.grid()
#     axes.tick_params(axis='x', labelsize=30)
#     axes.tick_params(axis='y', labelsize=30)
    
# fig.tight_layout() 
