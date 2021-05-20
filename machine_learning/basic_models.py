import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import pickle

"""
PROBLEMES A VERIFIER :
    - Consistance entre 'dataset_all' et 'dataset_event' pour les évènements
        - Tous les évènements présents dans All ?
        - Gestion des doublons ? (plusieurs casses)
"""

# General data
PATH = "data/"
# PATH = "../data/"
# Année de prédiction
A = 2008    # Fin de la période d'observation
P = 3       # Durée de la prédiction

# Preparing dataset
df = pd.read_csv(PATH + 'master_df_events.csv')
df_all = pd.read_csv(PATH + 'master_df_all.csv')

# Enlever les cases inconnues (uniquement valable pour le 1er dataset. Maintenant aucun inconnu
df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
df = df.drop(df[df.MATAGE == 'r'].index)
df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

# Building datetime features
df_all['DDP'] = pd.to_datetime(df_all['DDP'])
df_all['year_pose'] = df_all['DDP'].apply(lambda x: x.year)
df_all['DDCC'] = pd.to_datetime(df_all['DDCC'])
df_all['year_casse'] = df_all['DDCC'].apply(lambda x: x.year)
df['DDCC'] = pd.to_datetime(df['DDCC'])
df['year_casse'] = df['DDCC'].apply(lambda x: x.year)

# Enlever les casses des années >= A (on ne les connais pas encore)
df_all.loc[df_all.year_casse >= A] = np.NaN
df_all = df_all.drop_duplicates(keep = 'first') # On élimine les doublons : il se peut qu'un tuyau ait été cassé 2 fois dans une date > A, la seuls colonne qui les distingue est la date de casse (fixée à NaN) -> doublon

# Dupliquer les tuyaux cassés et les ré-inclure dans le dataset comme non cassé (de cette manière ils ne disparaissent pas)
# Les tuyaux réparés apparaitront dans le dataset comme existant, avec une réparation effectuée
# Prendre le dataset DF, trier dans l'ordre des ID et DDC
df = df.sort_values(['ID', 'DDCC'], ascending = [True, True])
# Enlever date de casse (tuyaux fonctionnel, mais ayant subit une réparation)
duplicate = df.loc[df.year_casse < A]
duplicate['DDCC'] = np.NaN
# Ne garder que les valeur unique de la colonne ID, avec l'option keep = last
duplicate = duplicate.drop_duplicates(subset ="ID", keep = 'last', inplace=False)
# Ajouter ce dataset à la suite de DF_ALL
df_all = pd.concat([df_all, duplicate], ignore_index=True, sort = False)


# Periode d'observation
obs = df_all.groupby(['collectivite'])['year_casse'].min().rename("obs_start").reset_index(drop=False)
df_all = pd.merge(df_all, obs, how = 'left', on='collectivite')
obs = df_all.groupby(['collectivite'])['year_casse'].max().rename("obs_end").reset_index(drop=False)
df_all = pd.merge(df_all, obs, how = 'left', on='collectivite')


# Date depuis derniere casse (traitement des tuyaux cassés plusieurs fois)
# Classement selon ID puis date de casse
df_all = df_all.sort_values(['ID', 'DDCC'], ascending = [True, True])
# Décallage ID et date de casse
df_all['ID2'] = df_all['ID'].shift(1).fillna(value=0)
df_all['DDCC2'] = df_all['year_casse'].shift(1).fillna(value=0)
# Définition par défaut de la date de dernière casse à la date de début de fenêtre
df_all['derniere_casse'] = df_all.obs_start
# Si jamais le même indice arrive 2 fois (il a été cassé 2 fois), alors on note la date de la derniere casse comme la date de casse de casse du même ID situé 1 case plus haut dans le tableau
df_all.loc[df_all.ID == df_all.ID2, "derniere_casse"] = df_all['DDCC2']
# On supprime les colonnes intermédiaires
df_all = df_all.drop(columns=['ID2', 'DDCC2'])

# Année de casse: on regarde chaque période d'observation comme une expérience, début = 1ere casse, fin = derniere casse
# Si le tuyau n'est pas cassé, on note simplement son "age" en fin de période d'observation.
# On précisera dans une variable 'event' (True / False) si il est cassé ou pas
df_all['event'] = True
df_all.loc[df_all.year_casse.isnull(), "event"] = False
df_all['year_casse'] = df_all['year_casse'].fillna(df_all['obs_end'])

# Calcul de la durée de vie: C'est une partie de la target, on veut que le modèle l'estime
df_all['duree_de_vie'] = df_all['year_casse'] - df_all['derniere_casse']
# df_all['TTF'] = df_all['duree_de_vie'] - A # durée de vie restante


# Nombre de réparations sur 1 tuyau (pour une date <= A)
count_reparation = df[df['year_casse'] < A].groupby(['ID']).size().rename("reparation").reset_index(drop=False)
df_all = pd.merge(df_all, count_reparation, how = 'left', on='ID')
df_all['reparation'] = df_all['reparation'].fillna(value = 0)


# Standardizing quantitative data
scaler = StandardScaler()
df_all['DIAMETRE'] = scaler.fit_transform(df_all[['DIAMETRE']])
df_all['year_pose'] = scaler.fit_transform(df_all[['year_pose']])


# Encoding quantitative variables : oneHotEncoding On supprime MATERIAU car l'info est contenue dans MATAGE
df_all = pd.concat([df_all, pd.get_dummies(df_all.collectivite)], axis = 1)
df_all = pd.concat([df_all, pd.get_dummies(df_all.MATAGE)], axis = 1)

# Target definition: Tuple: (Event, durée de vie) -> définie plus tard grâce à Surv.from_array
# df_all['target'] = df_all.apply(lambda row: (row.event, row.duree_de_vie), axis=1)


# New approach: At year A, we start a new experiment -> we test on all data with obs window fully included in A + P
# We learn on the whole dataset with all available data (failure & pose < A)
# We remove pipes that has not started observation period

learning_data = df_all[(df_all['year_pose'] < A) & (df_all['obs_start'] < A)]
learning_target = Surv.from_arrays(learning_data.event, learning_data.duree_de_vie, 'casse', 'durée de vie')
learning_data.index = learning_data.ID
learning_data = learning_data.drop(columns=['ID', 'DDCC', 'IDT', 'DDP', 'year_casse', 'event', 'duree_de_vie', 'obs_start', 'obs_end', 'derniere_casse', 'MATERIAU', 'MATAGE', 'collectivite'])

# Uniquement les non cassés
test_data = df_all[(df_all['year_pose'] < A) & (df_all['obs_start'] < A) & (df_all['obs_end'] > A+P) & (df_all['reparation'] == 0)]
test_target = Surv.from_arrays(test_data.event, test_data.duree_de_vie, 'casse', 'durée de vie')
test_data.index = test_data.ID
test_data = test_data.drop(columns=['ID', 'DDCC', 'IDT', 'DDP', 'year_casse', 'event', 'duree_de_vie', 'obs_start', 'obs_end', 'derniere_casse', 'MATERIAU', 'MATAGE', 'collectivite'])

# Model construction
random_state = 0
rsf = RandomSurvivalForest(n_estimators=500,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs=-1,
                           random_state=random_state)
rsf.fit(learning_data, learning_target)


# FIRST APPROACH : NOT OPTIMAL
# # Splitting train / test dataset
# # We want:
# #   - Shuffle dataset
# #   - 70% of events that happened before year A
# #   - Try to keep proportion equality amongst collectivity between train & test
# #   - Add 70% of pipes not broken to fit & the rest to test
#
# p_train = 0.70
# # Start from the event dataset
# df_temp = learning_data[learning_data.reparation != 0]
# IDs_event = df_temp.ID
# # Nombres d'event par collectivités
# print(df_temp.collectivite.value_counts())
# # Grouper les 3 collectivités avec le moins de cas (19-20-24) car trop peu d'évènements
# union = list(df_temp.collectivite.value_counts()[-3:].index)
# df_temp.loc[df_temp.collectivite.isin(union), "collectivite"] = 'Autres'
# # Dataset of events
# x_train, x_test, y_train, y_test = train_test_split(df_temp.iloc[:, :-1], df_temp.iloc[:,-1], test_size=1-p_train, shuffle = True, stratify = df_temp.collectivite, random_state=0)
# # Dataset of clean pipes
# df_temp = learning_data[learning_data.reparation == 0]
# df_temp.loc[df_temp.collectivite.isin(union), "collectivite"] = 'Autres'
# x_train2, x_test2, y_train2, y_test2 = train_test_split(df_temp.iloc[:, :-1], df_temp.iloc[:,-1], test_size=1-p_train, shuffle = True, stratify = df_temp.collectivite, random_state=0)
# # Association
# x_train = pd.concat([x_train, x_train2])
# y_train = pd.concat([y_train, y_train2])
# x_test = pd.concat([x_test, x_test2])
# y_test = pd.concat([y_test, y_test2])


