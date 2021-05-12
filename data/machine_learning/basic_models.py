import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

"""
PROBLEMES A VERIFIER :
    - Consistance entre 'dataset_all' et 'dataset_event' pour les évènements
        - Tous les évènements présents dans All ?
        - Evenements en double ?
    
    - Fitting / Test
        - Dans le dataset de test, éliminer les redondances: les tuyaux cassés plusieurs fois = on ne garde que la dernière fois

"""

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
df_all['year_casse'] = df_all['year_casse'].fillna(value=A)
df['DDCC'] = pd.to_datetime(df['DDCC'])
df['year_casse'] = df['DDCC'].apply(lambda x: x.year)


# Date depuis derniere casse
# Classement selon ID puis date de casse
df_all = df_all.sort_values(['ID', 'DDCC'], ascending = [True, True])
# Décallage ID et date de casse
df_all['ID2'] = df_all['ID'].shift(1).fillna(value=0)
df_all['DDCC2'] = df_all['year_casse'].shift(1).fillna(value=0)
# Définition par défaut de la date de dernière casse à la date de pose
df_all['derniere_casse'] = df_all.year_pose
# Si jamais le même indice arrive 2 fois (il a été cassé 2 fois), alors on note la date de la derniere casse comme la date de casse de casse du même ID situé 1 case plus haut dans le tableau
df_all.loc[df_all.ID == df_all.ID2, "derniere_casse"] = df_all['DDCC2']
# On supprime les colonnes intermédiaires
df_all = df_all.drop(columns=['ID2', 'DDCC2'])


# Calcul de la durée de vie
df_all['duree_de_vie'] = df_all['year_casse'] - df_all['derniere_casse'] # C'est une partie de la target: on veut que le modèle l'estime
# df_all['TTF'] = df_all['duree_de_vie'] - A # durée de vie restante


# Nombre de réparations sur 1 tuyau (pour une date <= A)
count_reparation = df[df['year_casse'] <= A].groupby(['ID']).size().rename("reparation").reset_index(drop=False)
df_all = pd.merge(df_all, count_reparation, how = 'left', on='ID')
df_all['reparation'] = df_all['reparation'].fillna(value = 0)


# Standardizing quantitative data
scaler = StandardScaler()
df_all['DIAMETRE'] = scaler.fit_transform(df_all[['DIAMETRE']])
df_all['year_pose'] = scaler.fit_transform(df_all[['year_pose']])


# Encoding quantitative variables :
# Using decision trees so can handle categorical variables

# Target definition: Tuple: (Event, durée de vie)
df_all['event'] = False
df_all.loc[df_all.reparation != 0, "event"] = True
df_all['target'] = df_all.apply(lambda row: (row.event, row.duree_de_vie), axis=1)

# Learning data -> selecting existing pipes & removing unnecessary columns
learning_data = df_all[df_all['year_pose'] <= A]
learning_data = learning_data.drop(columns=['DDCC', 'IDT', 'DDP', 'year_casse', 'event', 'duree_de_vie'])


# Splitting train / test dataset
# We want:
#   - Shuffle dataset
#   - 70% of events that happened before year A
#   - Try to keep proportion equality amongst collectivity between train & test
#   - Add 70% of pipes not broken to fit & the rest to test

p_train = 0.70
# Start from the event dataset
df_temp = learning_data[learning_data.reparation != 0]
# Nombres d'event par collectivités
print(df_temp.collectivite.value_counts())
# Grouper les 3 collectivités avec le moins de cas (19-20-24) car trop peu d'évènements
union = list(df_temp.collectivite.value_counts()[-3:].index)
df_temp.loc[df_temp.collectivite in union, "collectivite"] = 'Autres'

x_train, x_test, y_train, y_test = train_test_split(df_temp.iloc[:, :-1], df_temp.iloc[:,-1], test_size=1-p_train, shuffle = True, stratify = df_temp.collectivite, random_state=0)


# Building model
rf = RandomForestClassifier(n_estimators = 1000, max_depth = None, random_state=0)







