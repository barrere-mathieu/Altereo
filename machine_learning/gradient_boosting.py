import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
import pickle
import matplotlib.pyplot as plt
from machine_learning import calcul_AUC


# General parameters
PATH = "../data/"
A = 2010    # Année de prédiction
INIT_TIME = 20  # Initialisation du modèle: date de dernière casse prises au hasard
# Dataset load
DATASET_LOAD = True
DATASET_NAME = 'data_prep_2010.csv'
MODEL_FIT = True # Set to false if you want to load a model
MODEL_NAME = 'XGB_1' # Set to None if you don't want to save the model
MODEL_LOAD = MODEL_NAME + ".sav" # Path to the model to load. Only relevent if MODEL_FIT is False.


def calcul_periode_obs(d):
    obs = d.groupby(['collectivite'])['year_casse'].min().rename("obs_start").reset_index(drop=False)
    d = pd.merge(d, obs, how='left', on='collectivite')
    obs = d.groupby(['collectivite'])['year_casse'].max().rename("obs_end").reset_index(drop=False)
    d = pd.merge(d, obs, how='left', on='collectivite')
    return d

def calcul_derniere_casse(df, init = 0):
    # Classement selon ID puis date de casse
    df = df.sort_values(['ID', 'DDCC'], ascending=[True, True])
    # Décallage ID et date de casse
    df['ID2'] = df['ID'].shift(1).fillna(value=0)
    df['DDCC2'] = df['year_casse'].shift(1).fillna(value=0)
    # Définition par défaut de la date de dernière casse à la date de début de fenêtre - 20 an (ou date de pose si elle arrive plus tard)
    # df['temp'] = df.obs_start - init
    # df['derniere_casse'] = df[['year_pose', 'temp']].max(axis=1)
    df['derniere_casse'] = np.NaN
    df['temp'] = df.obs_start - init
    df['temp2'] = df.obs_start - int(2*init)
    df.loc[df.year_pose <= df.temp2, 'derniere_casse'] = df.loc[df.year_pose <= df.temp2, ['year_pose', 'temp']].max(axis=1)
    df.derniere_casse = df.derniere_casse.fillna(df['year_pose'])
    # Si jamais le même indice arrive 2 fois (il a été cassé 2 fois), alors on note la date de la derniere casse comme la date de casse de casse du même ID situé 1 case plus haut dans le tableau
    df.loc[df.ID == df.ID2, "derniere_casse"] = df['DDCC2']
    # On supprime les colonnes intermédiaires
    df = df.drop(columns=['ID2', 'DDCC2', 'temp', 'temp2'])
    return df


def prep_dataset(path, censure = A, init = 0):
    # Preparing dataset
    print('Setting up datasets')
    df = pd.read_csv(path + 'master_df_events.csv')
    df_all = pd.read_csv(path + 'master_df_all.csv')

    # Enlever les cases inconnues (uniquement valable pour le 1er dataset)
    df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
    df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
    df = df.drop(df[df.MATAGE == 'r'].index)
    df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

    # Building datetime features
    df_all['DDP'] = pd.to_datetime(df_all['DDP'])
    df_all['year_pose'] = df_all['DDP'].apply(lambda x: x.year)
    df_all['DDCC'] = pd.to_datetime(df_all['DDCC'])
    df_all['year_casse'] = df_all['DDCC'].apply(lambda x: x.year)
    df['DDP'] = pd.to_datetime(df['DDP'])
    df['year_pose'] = df['DDP'].apply(lambda x: x.year)
    df['DDCC'] = pd.to_datetime(df['DDCC'])
    df['year_casse'] = df['DDCC'].apply(lambda x: x.year)

    periode_obs = pd.merge(
        df_all.groupby(['collectivite'])['year_casse'].min().rename("obs_start").reset_index(drop=False),
        df_all.groupby(['collectivite'])['year_casse'].max().rename("obs_end").reset_index(drop=False),
        on="collectivite")

    # Enlever les tuyaux posés après A (ils ne sont pas encore installés)
    df_all = df_all.loc[(df_all['year_pose'] < censure)]

    # Enlever les casses des années >= A (on ne les connait pas encore)
    df_all.loc[df_all.year_casse >= censure, 'DDCC'] = np.NaN
    df_all.loc[df_all.year_casse >= censure, 'year_casse'] = np.NaN

    # Dupliquer les tuyaux cassés et les ré-inclure dans le dataset comme non cassé (de cette manière ils ne disparaissent pas)
    # Les tuyaux réparés apparaitront dans le dataset comme existant, avec une réparation effectuée
    # Prendre le dataset DF, trier dans l'ordre des ID et DDC
    df = df.sort_values(['ID', 'DDCC'], ascending = [True, True])
    # Enlever date de casse (tuyaux fonctionnel, mais ayant subit une réparation)
    duplicate = df.loc[(df.year_casse < censure) & (df.year_pose < censure)]
    duplicate['DDCC'] = np.NaN
    duplicate['year_casse'] = np.NaN
    # Ne garder que les valeur unique de la colonne ID, avec l'option keep = last
    duplicate = duplicate.drop_duplicates(subset ="ID", keep = 'last', inplace=False)
    # Ajouter ce dataset à la suite de DF_ALL
    df_all = pd.concat([df_all, duplicate], ignore_index=True, sort = False)
    df_all = df_all.drop_duplicates(keep = 'first') # On élimine les doublons : il se peut qu'un tuyau ait été cassé 2 fois dans une date > A, la seuls colonne qui les distingue est la date de casse (fixée à NaN) -> doublon

    # Calcul periode d'observation
    # df_all = calcul_periode_obs(df_all)
    df_all = pd.merge(df_all, periode_obs, how='left', on='collectivite')

    # Calcul date depuis derniere casse (traitement des tuyaux cassés plusieurs fois)
    df_all = calcul_derniere_casse(df_all, init = init)

    # Année de casse: on regarde chaque période d'observation comme une expérience, début = 1ere casse, fin = derniere casse
    # Si le tuyau n'est pas cassé, on note simplement son "age" en fin de période d'observation.
    # On précisera dans une variable 'event' (True / False) si il est cassé ou pas
    df_all['event'] = True
    df_all.loc[df_all.year_casse.isnull(), "event"] = False
    df_all['temp'] = censure-1
    df_all['temp'] = df_all[['obs_end', 'temp']].min(axis=1)
    df_all['year_casse'] = df_all['year_casse'].fillna(df_all['temp'])
    df_all = df_all.drop(columns=['temp'])

    # Calcul de la durée de vie: C'est une partie de la target, on veut que le modèle l'estime
    df_all['duree_de_vie'] = df_all['year_casse'] - df_all['derniere_casse']

    # On supprime les tuyaux étant installés APRES la fin d'une période d'observation (durée de vie négative car : end_obs < year_pose <= censure)
    # Ces tuyaux ne nous sont pas utiles -> on n'apprendra rien d'eux et on ne test que sur les collectivités vérifiants end_obs > censure
    df_all = df_all.loc[(df_all['duree_de_vie'] >= 0)]

    # Nombre de réparations sur 1 tuyau (pour une date <= A)
    count_reparation = df[df['year_casse'] < censure].groupby(['ID']).size().rename("reparation").reset_index(drop=False)
    df_all['reparation'] = 0
    df_all.loc[(df_all['reparation'] == 0) & (df_all["year_pose"] <= df_all['obs_start'] - 2*init), "reparation"] = 1 # Initialisation à 1 pour les tuyaux plus vieux que l'initialisation
    for id in count_reparation.ID:
        temp = df_all.loc[df_all['ID'] == id]
        init_reparation = list(temp['reparation'])[0]
        for i, j in enumerate(temp.index):
            df_all.loc[j, "reparation"] = i + init_reparation

    df_all.to_csv(PATH + 'data_prep_' + str(censure) + '.csv')

    return df_all


def prep_target_dataset(path, init = 0, prediction = A):
    # Preparing dataset
    print('Setting up target')
    df = pd.read_csv(path + 'master_df_events.csv')
    df_all = pd.read_csv(path + 'master_df_all.csv')

    # Enlever les cases inconnues (uniquement valable pour le 1er dataset)
    df = df.drop(df[df.MATERIAU == 'INCONNU'].index)
    df_all = df_all.drop(df_all[df_all.MATERIAU == 'INCONNU'].index)
    df = df.drop(df[df.MATAGE == 'r'].index)
    df_all = df_all.drop(df_all[df_all.MATAGE == 'r'].index)

    # Building datetime features
    df_all['DDP'] = pd.to_datetime(df_all['DDP'])
    df_all['year_pose'] = df_all['DDP'].apply(lambda x: x.year)
    df_all['DDCC'] = pd.to_datetime(df_all['DDCC'])
    df_all['year_casse'] = df_all['DDCC'].apply(lambda x: x.year)
    df['DDP'] = pd.to_datetime(df['DDP'])
    df['year_pose'] = df['DDP'].apply(lambda x: x.year)
    df['DDCC'] = pd.to_datetime(df['DDCC'])
    df['year_casse'] = df['DDCC'].apply(lambda x: x.year)

    periode_obs = pd.merge(
        df_all.groupby(['collectivite'])['year_casse'].min().rename("obs_start").reset_index(drop=False),
        df_all.groupby(['collectivite'])['year_casse'].max().rename("obs_end").reset_index(drop=False),
        on="collectivite")

    # Dupliquer les tuyaux cassés et les ré-inclure dans le dataset comme non cassé (de cette manière ils ne disparaissent pas)
    # Les tuyaux réparés apparaitront dans le dataset comme existant, avec une réparation effectuée
    # Prendre le dataset DF, trier dans l'ordre des ID et DDC
    df = df.sort_values(['ID', 'DDCC'], ascending = [True, True])
    duplicate = df.loc[df.year_pose < prediction]
    duplicate['DDCC'] = np.NaN
    duplicate['year_casse'] = np.NaN
    duplicate = duplicate.drop_duplicates(subset ="ID", keep = 'last', inplace=False)
    df_all = pd.concat([df_all, duplicate], ignore_index=True, sort = False)

    # Calcul periode d'observation
    df_all = pd.merge(df_all, periode_obs, how='left', on='collectivite')

    # Calcul date depuis derniere casse (traitement des tuyaux cassés plusieurs fois)
    df_all = calcul_derniere_casse(df_all, init = init)

    # Année de casse
    df_all['event'] = 1
    df_all.loc[df_all.year_casse.isnull(), "event"] = 0
    df_all['year_casse'] = df_all['year_casse'].fillna(df_all['obs_end'])
    # Calcul de la durée de vie: C'est une partie de la target, on veut que le modèle l'estime
    df_all['duree_de_vie'] = df_all['year_casse'] - df_all['derniere_casse']

    # Define unique ID
    df_all['ID_u'] = df_all['ID'] + df_all['derniere_casse'].astype(str)
    df_all.index = df_all.ID_u
    df_all.index.name = None
    return df_all

def plot_surv(surv_curves, k, display):
    np.random.shuffle(surv_curves)
    for i, s in enumerate(surv_curves[:k, :]):
        plt.step(model.event_times_, s, where="post", label=str(i))
    plt.ylabel("Survival probability")
    plt.xlabel("Time in years")
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_NAME + '_surv_' + str(k) + '.png')

    if display:
        plt.show()


class EarlyStoppingMonitor:

    def __init__(self, window_size, max_iter_without_improvement):
        self.window_size = window_size
        self.max_iter_without_improvement = max_iter_without_improvement
        self._best_step = -1

    def __call__(self, iteration, estimator, args):
        # continue training for first self.window_size iterations
        if iteration < self.window_size:
            return False

        # compute average improvement in last self.window_size iterations.
        # oob_improvement_ is the different in negative log partial likelihood
        # between the previous and current iteration.
        start = iteration - self.window_size + 1
        end = iteration + 1
        improvement = np.mean(estimator.oob_improvement_[start:end])

        if improvement > 1e-6:
            self._best_step = iteration
            return False  # continue fitting

        # stop fitting if there was no improvement
        # in last max_iter_without_improvement iterations
        diff = iteration - self._best_step
        return diff >= self.max_iter_without_improvement




#----------------------------------------------------------------------------------------------------------------------#
# MAIN

if DATASET_LOAD:
    print('Loading Dataset')
    df_all = pd.read_csv(PATH + DATASET_NAME, index_col=0)
else:
    df_all = prep_dataset(PATH, init=INIT_TIME, censure=A)

# Standardizing quantitative data
scaler = StandardScaler()
df_all['DIAMETRE_std'] = scaler.fit_transform(df_all[['DIAMETRE']])
df_all['year_pose_std'] = scaler.fit_transform(df_all[['year_pose']])

# Encoding quantitative variables : oneHotEncoding On supprime MATERIAU car l'info est contenue dans MATAGE
df_all = pd.concat([df_all, pd.get_dummies(df_all.collectivite)], axis = 1)
df_all = pd.concat([df_all, pd.get_dummies(df_all.MATAGE)], axis = 1)

# New approach: At year A, we start a new experiment -> we test on all data with obs window fully included in A + P
# We learn on the whole dataset with all available data (failure & pose < A)
# We remove pipes that has not started observation period

learning_data = df_all[(df_all['year_pose'] < A) & (df_all['obs_start'] < A)]
learning_data = learning_data.loc[:, (learning_data != 0).any(axis=0)]
learning_target = Surv.from_arrays(learning_data.event, learning_data.duree_de_vie, 'casse', 'duree_de_vie')
learning_data.index = learning_data.ID
learning_data = learning_data.drop(columns=['ID', 'DDCC', 'IDT', 'DDP', 'year_casse', 'event', 'duree_de_vie', 'obs_start', 'obs_end', 'MATERIAU', 'MATAGE', 'collectivite', 'DIAMETRE', 'year_pose'])

# Test data : tous les membre du réseau étant dans une fenêtre d'observation active (tuyaux non cassés car tuyaux cassés sont remplacés)
test_data = df_all[(df_all['year_pose'] < A) & (df_all['obs_start'] < A) & (df_all['obs_end'] > A) & (df_all.DDCC.isnull())]
test_data.index = test_data.ID
test_data = test_data[list(learning_data.columns)]

# Model construction
if MODEL_FIT:
    print('Fitting model')
    random_state = 0
    model = GradientBoostingSurvivalAnalysis(n_estimators=1000,
                               learning_rate=0.05,
                               max_depth = 1,
                               subsample = 0.5,
                               dropout_rate=0,
                               max_features=None, # test sqrt
                               random_state=random_state,
                               njob = 50)
    monitor = EarlyStoppingMonitor(25, 50)
    model.fit(learning_data, learning_target, monitor=monitor)
    print("Fitted base learners:", model.n_estimators_)

    if MODEL_NAME is not None:
        pickle.dump(model, open(MODEL_NAME + '.sav', 'wb'))
else:
    print('Loading model')
    model = pickle.load(open(MODEL_LOAD, 'rb'))
print('Model predicting')
pred = model.predict(test_data)

# Association prediction with real labels
print('Aggregating results')
# Redéfinir test data
test_data['ID_u'] = test_data.index + test_data['derniere_casse'].astype(str)
test_data.index = test_data['ID_u']
test_data['proba'] = pred
test_data.index.name = None

# Gathering real test events
temp = prep_target_dataset(PATH, init = INIT_TIME, prediction=A)
temp = temp.drop_duplicates(subset ="ID_u", keep = 'first', inplace=False)

# Combining prediction and events
test_data = pd.concat([test_data, temp['event']], axis = 1, join='inner')
test_data = pd.concat([test_data, temp['duree_de_vie']], axis = 1, join='inner')
test_target = Surv.from_arrays(test_data.event, test_data.duree_de_vie, 'casse', 'duree_de_vie')

# Plotting AUC dynamic
times = np.unique(np.percentile(test_target["duree_de_vie"], np.linspace(5, 81, 10)))
calcul_AUC.plot_cumulative_dynamic_auc(test_data, times, MODEL_NAME, display=False)

# Ploting standard ROC curve
annotations = [
    ('Model', 'Random Survival Forest'),
    ('Année prédiction', A),
    ('Nombre de collectivités', len(temp.collectivite.unique())),
    ('Nombre tuyaux total', test_data.shape[0]),
    ('Nombre de casses', sum(test_data.event)),
]
calcul_AUC.save_AUC(test_data, MODEL_NAME, annotations)
