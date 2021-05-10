import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# General data
PATH = "../data/"
# Année de prédiction
A = 2008

# Preparing dataset
df = pd.read_csv(PATH + 'master_df_events.csv')
df_all = pd.read_csv(PATH + 'master_df_all.csv')

df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

# Building datetime features
df_all['DDP'] = pd.to_datetime(df_all['DDP'])
df_all['year_pose'] = df_all['DDP'].apply(lambda x: x.year)
df_all['DDCC'] = pd.to_datetime(df_all['DDCC'])
df_all['year_casse'] = df_all['DDCC'].apply(lambda x: x.year)
df_all['year_casse'] = df_all['year_casse'].fillna(value=A+100)
df['DDCC'] = pd.to_datetime(df['DDCC'])
df['year_casse'] = df['DDCC'].apply(lambda x: x.year)

# Calcul de la durée de vie
df_all['duree_de_vie'] = df_all['year_casse'] - df_all['year_pose'] # Peut etre pas utile -> c'est la target: on veut que le modèle aprenne dessus
df_all['TTF'] = df_all['duree_de_vie'] - A # durée de vie restante

# Nombre de réparations sur 1 tuyau
count_reparation = df[df['year_casse'] <= A].groupby(['ID']).size().rename("reparation").reset_index(drop=False)
df_all = pd.merge(df_all, count_reparation, on=['ID'])
df_all['reparation'] = df_all['reparation'].fillna(value = 0)

# Date depuis derniere casse
df_all = df_all.sort_values(['ID', 'DDCC'], ascending = [True, True])
df_all['ID2'] = df_all['ID'].shift(1).fillna(value=0)
df_all['DDCC2'] = df_all['year_casse'].shift(1).fillna(value=0)
df_all['derniere_casse'] = df_all.year_pose
df_all.loc[df_all.ID == df_all.ID2, "derniere_casse"] = df_all['DDCC2']
df_all = df_all.drop(columns=['ID2', 'DDCC2'])

# Encoding quantitative variables :
# Using decision trees so can handle categorical variables

# Target definition:
#   2 targets:
#         - simple: 0/1: cassé / non cassé
#         - complexe: time to failure (durée de vie - année d'observation) -> implémenter algorithme de classement: ce n'est pas la valeur absolue mais la valeur relative qui nous intéresse
df_all['y1'] = 0
df_all.loc[df_all.DDCC.isna() == False, "y1"] = 1

# Learning data
learning_data = df_all[df_all['year_pose'] <= A]
learning_data = learning_data.drop(columns=['DDCC', 'IDT', 'DDP', 'ID', 'year_casse', 'duree_de_vie'])

# Standardizing quantitative data
scaler = StandardScaler()
learning_data['DIAMETRE'] = scaler.fit_transform(learning_data[['DIAMETRE']])
learning_data['year_pose'] = scaler.fit_transform(learning_data[['year_pose']])

# Building model
rf = RandomForestClassifier(n_estimators = 1000, max_depth = None, random_state=0)







