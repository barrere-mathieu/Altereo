import pandas as pd
import numpy as np
import math 
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots


import plotly.graph_objs as go
from plotly import subplots
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# Path to data
PATH = "data/"

df = pd.read_csv(PATH + "master_df_events.csv")
df_all = pd.read_csv(PATH + 'master_df_all.csv')

df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

# nombre de casses en fonction date de casse
df['DDCC'] = pd.to_datetime(df['DDCC'])
df['year_event'] = df['DDCC'].apply(lambda x: x.year)
group = df.groupby(['collectivite', 'year_event']).size().reset_index()
group_all =df_all.groupby(["collectivite"]).size().reset_index()
group = group.rename(index = str, columns={0 : "nb_casses"})


# récupérer les données de la collectivité 22
tab_AS = group[group.collectivite == "Collectivite_22"]


#calcul de ti
tab_AS["year_decal"] = group["year_event"]
tab_AS.year_decal = tab_AS.year_decal.shift(periods=1)
tab_AS = tab_AS.replace(np.nan, group.year_event[155])
tab_AS["ti"] = group["year_event"] - tab_AS["year_decal"]
tab_AS = tab_AS.drop(["year_decal"], axis = 1)

#calcul de ni: nombre de tuyaux qu'on a à l'instant T0 (les indivudus à traiter)
nb_ty = group_all[group_all.collectivite == "Collectivite_22"].values[0,1]
#tab_AS2 = tab_AS.rename(index = str, columns={"0" : "nb_casses",})

print(tab_AS.columns)

# tab_AS["nb_Tuyaux"] = nb_ty - 
# tab_AS["nb_Tuyaux"]  = 
# tab_AS.nb_Tuyaux = tab_AS.nb_Tuyaux.shift(periods=1)
# tab_AS.nb_Tuyaux = tab_AS.nb_Tuyaux.fillna(nb_ty)


print(tab_AS["nb_Tuyaux"].iloc[0])  
    
    
    
    
    
    
    
    
    
