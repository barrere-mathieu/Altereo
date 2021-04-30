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

nb_pannes= df.groupby(["collectivite"]).size()
nb_collect = nb_pannes.index #df.groupby(["collectivite"]).collectivite.unique()
Diam = df.groupby(["collectivite"]).DIAMETRE.mean()
MAteriau = df.groupby(["collectivite"]).MATERIAU.unique()


fig = px.scatter(df, x=Diam, y=nb_collect, size=nb_pannes,
                 size_max=40, color =Diam)
	       
fig.show()






