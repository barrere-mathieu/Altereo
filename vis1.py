# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:42:27 2021

@author: meksi
"""
import pandas as pd
import numpy as np
import os
from itertools import combinations

import plotly as py
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"


df = pd.read_csv("data/master_df_events.csv")
df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)

# Date  = [each.replace('/', '-') for each in df.DDP]
# df['DDP'] = Date
# df['DDP'].str.extract(r'.*(\d\d-\d\d/-d\d\d\d)')
# df["DDP"] = pd.to_datetime(df["DDP"], format="%d-%m-%Y").dt.date

test = df.groupby(["collectivite"]).MATERIAU.unique().values[:]
nb_pannes= df.groupby(["collectivite"]).size()
nb_collect = nb_pannes.index #df.groupby(["collectivite"]).collectivite.unique()
Diam = df.groupby(["collectivite"]).DIAMETRE.mean()
MAteriau = df.groupby(["collectivite"]).MATERIAU.unique()
 
print(MAteriau.values[:])


fig = px.scatter(df, x=nb_pannes, y=nb_collect, size=nb_pannes, 
                 size_max=40, color =Diam)
	       
fig.show()






