import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import datetimes_to_durations
from lifelines.statistics import multivariate_logrank_test
import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error, accuracy_score


from matplotlib import pyplot as plt
import plotly.io as pio
pio.renderers.default = "browser"
import matplotlib as mpl
from matplotlib.offsetbox import AnchoredText


def calcul_periode_obs(d):
    obs = d.groupby(['collectivite'])['year_event'].min().rename("obs_start").reset_index(drop=False)
    d = pd.merge(d, obs, how='left', on='collectivite')
    obs = d.groupby(['collectivite'])['year_event'].max().rename("obs_end").reset_index(drop=False)
    d = pd.merge(d, obs, how='left', on='collectivite')
    return d

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

# Periode d'observation
df_all = calcul_periode_obs(df_all)
df_all['limit_low'] = df_all['obs_start'] - 20


# les tuyaux installés dans l'intervalle [debut obs - 20 ans; end obs]
df1 = df_all.loc[(df_all['year_pose'] >= df_all['limit_low']) & (df_all['year_pose'] < df_all['obs_end'])]
# df1 = df_all.drop(df_all[df_all.year_pose < df_all['limit_low']].index)
# df1 = df1.drop(df1[df1.year_pose >= 2017].index)
# #par curiosité:
# df_t1 = df1.groupby(df1.ID).size().reset_index().rename(columns = {0: "nb_casse"})
# a = df_t1.nb_casse.max()
# df_t1 = df_t1.drop(df_t1[df_t1.nb_casse < 3].index)

# les tuyaux installés n'importe quand avec 2 fois de casses pendant [debut obs; end obs]
df2 = df_all.loc[df_all['year_pose'] < df_all['limit_low']]
df2 = df2.loc[(df2.year_event >=df2['limit_low']) & (df2.year_pose < df2.obs_end)]
#df2 = df_all.drop(df_all[df_all.year_pose > 1950].index)
df_t = df2.groupby(df2.ID).size().reset_index().rename(columns = {0: "nb_casse"})
#a = df_t.nb_casse.max()
df_t = df_t.drop(df_t[df_t.nb_casse < 2].index)
df2 = df2.merge(df_t, how= 'inner', on ='ID').drop(columns = ['nb_casse'])


#regrouper les 2 catégories des tuyaux
df_T = pd.concat([df2, df1], ignore_index=True)


# sélection des tuyaux cassés pour la regression
df_casse = df_T.dropna(subset = ["DDCC"])


# 2ème étape est de calculer la durée de vie pour chaque tuyau
df_casse = df_casse.sort_values(['ID', 'DDCC'], ascending = [True, True])
df_casse['ID2'] = df_casse['ID'].shift(1).fillna(value=0)
df_casse['DDCC2'] = df_casse['year_event'].shift(1).fillna(value=0)
# Si jamais le même indice arrive plusieurs fois (il a été cassé 2 fois), alors on note la date de la derniere casse comme la date de casse du même ID situé 1 case plus haut dans le tableau
df_casse.loc[df_casse.ID == df_casse.ID2, "derniere_casse"] = df_casse['DDCC2']
#durée de vie
df_casse['duree_de_vie'] = df_casse['year_event'] - df_casse['year_pose']
df_casse.loc[df_casse.ID == df_casse.ID2, "duree_de_vie"] = df_casse['year_event'] - df_casse['derniere_casse']
# On supprime les colonnes intermédiaires, matériau et les tuyaux qui on une durée de vie > 63 ans
df_casse = df_casse.drop(columns=['derniere_casse','ID2', 'DDCC2'])
df_casse = df_casse.drop(df_casse[df_casse.duree_de_vie > 63].index)
#df_test = df_casse.groupby(df_casse.MATAGE).size()
df_casse = df_casse.drop(df_casse[df_casse.MATAGE == 'ACIER19201940'].index)
df_casse = df_casse.drop(df_casse[df_casse.MATAGE == 'BONNA19651975'].index)
#df_casse = df_casse.drop(df_casse[df_casse.duree_de_vie == 0].index)



# Standardizing quantitative data
scaler = StandardScaler()
df_casse['DIAMETRE_std'] = scaler.fit_transform(df_casse[['DIAMETRE']])
df_casse['year_pose_std'] = scaler.fit_transform(df_casse[['year_pose']])
df_casse = df_casse.drop(columns=['DIAMETRE','IDT'])

# Encoding quanlitative variables : oneHotEncoding On supprime MATERIAU car l'info est contenue dans MATAGE
df_casse = pd.concat([df_casse, pd.get_dummies(df_casse.collectivite)], axis = 1)
df_casse = pd.concat([df_casse, pd.get_dummies(df_casse.MATAGE)], axis = 1)

#splittting Data
p_train = 0.70
Y = df_casse.duree_de_vie
X = df_casse.drop(columns=['ID', 'limit_low', 'DDCC', 'obs_start', 'obs_end', 'DDP', 'year_event', 'collectivite', 'MATERIAU', 'duree_de_vie', 'year_pose'])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1-p_train, shuffle = True, stratify = X.MATAGE, random_state=0)
x_train = x_train.drop(columns = ['MATAGE'])
x_test = x_test.drop(columns = ['MATAGE'])


# regression qu'on commence par un gridsearch pour sélectionner les best parametres
tuned_parameters = {'C':[.1, 10, 100, 1000]}

clf = GridSearchCV(SVR(kernel = 'rbf'), tuned_parameters, cv = 5, scoring = 'precision')
clf.fit(x_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_estimator_)
regressor = clf.best_estimator_


y_pred = regressor.predict(x_test)

#evalution avec score
mse = np.sqrt(mean_squared_error(y_test, y_pred))
R = regressor.score(x_train, y_train)

# tracer les résultats prédict en fonction des résultats observés

fig = px.scatter(group, x="diametre_range", y="year_pose_range", size=group["nb_casses"],
                 size_max=40, color=group["nb_casses"])

fig.show()



######### estimer une durée de vie et une date de casse pour les tuyaux sélectionnés
# sélectionner que les tuyaux qui n'ont pas une date de casse
df_Non_casse = df_T.loc[~df_T.index.isin(df_T.dropna().index)]

# Standardizing quantitative data
df_Non_casse['DIAMETRE_std'] = scaler.fit_transform(df_Non_casse[['DIAMETRE']])
df_Non_casse['year_pose_std'] = scaler.fit_transform(df_Non_casse[['year_pose']])

# Encoding quantitative variables : oneHotEncoding On supprime MATERIAU car l'info est contenue dans MATAGE
df_Non_casse = pd.concat([df_Non_casse, pd.get_dummies(df_Non_casse.collectivite)], axis = 1)
df_Non_casse = pd.concat([df_Non_casse, pd.get_dummies(df_Non_casse.MATAGE)], axis = 1)

#supprimer les colonnes inutiles
df_Non_casse = df_Non_casse.drop(columns=['DIAMETRE','IDT', 'obs_start', 'obs_end', 'ID', 'DDCC', 'collectivite', 'MATAGE', 'DDP', 'year_event', 'year_pose'])

#appliquer le modèle de regression
y_pred_non_casse = regressor.predict(df_Non_casse)
















# # 2ème étape est de calculer la durée de vie
# df1 = df1.sort_values(['ID', 'DDCC'], ascending = [True, True])
# df1['ID2'] = df1['ID'].shift(1).fillna(value=0)
# df1['DDCC2'] = df1['year_event'].shift(1).fillna(value=0)
# #df1['derniere_casse'] = df1.year_event
# # Si jamais le même indice arrive plusieurs fois (il a été cassé 2 fois), alors on note la date de la derniere casse comme la date de casse du même ID situé 1 case plus haut dans le tableau
# df1.loc[df1.ID == df1.ID2, "derniere_casse"] = df1['DDCC2']
# #durée de vie
# df1['duree_de_vie'] = df1['year_event'] - df1['year_pose']
# df1.loc[df1.ID == df1.ID2, "duree_de_vie"] = df1['year_event'] - df1['derniere_casse']
# # On supprime les colonnes intermédiaires
# df1 = df1.drop(columns=['ID2', 'DDCC2'])
# #a = df1.duree_de_vie.max()


# # 2ème étape est de calcumler la durée de vie pour chaque tuyau
# df2 = df2.sort_values(['ID', 'DDCC'], ascending = [True, True])
# df2['ID2'] = df2['ID'].shift(1).fillna(value=0)
# df2['DDCC2'] = df2['year_event'].shift(1).fillna(value=0)
# # Si jamais le même indice arrive plusieurs fois (il a été cassé 2 fois), alors on note la date de la derniere casse comme la date de casse du même ID situé 1 case plus haut dans le tableau
# df2.loc[df2.ID == df2.ID2, "derniere_casse"] = df2['DDCC2']
# #durée de vie
# df2['duree_de_vie'] = df2['year_event'] - df2['year_pose']
# df2.loc[df2.ID == df2.ID2, "duree_de_vie"] = df2['year_event'] - df2['derniere_casse']
# # On supprime les colonnes intermédiaires et les tuyaux qui on une durée de vie > 63 ans
# df2 = df2.drop(columns=['ID2', 'DDCC2'])
# df2 = df2.drop(df2[df2.duree_de_vie > df1.duree_de_vie.max()].index)


# ## compute AUROC and ROC curve values
# # prediction probabilities
# r_prob = [0 for _ in range (len(y_test))]
# rSVR_prob = regressor.predict_proba(x_test)

# r_prob = r_prob[:,1]
# rSVR_prob = rSVR_prob[:,1<<<<<<<<<<<<]

# # calculate AUROC
# r_auc = roc_auc_score(y_test, r_prob)
# rSVR_auc = roc_auc_score(y_test, rSVR_prob)

# #calculate ROC curve
# r_fpr, r_tpr, _ = roc_curve(y_test, r_prob)
# rsvr_fpr, rsvr_tpr, _ = roc_curve(y_test, rSVR_prob)

# #plot the ROC 
# plt.plot(r_fpr, r_tpr, linestyle = '--', label = 'Random prediction (AUROC = %0.2f)' %r_auc )
# plt.plot(rsvr_fpr, rsvr_tpr, marker = '.', label = 'SVR prediction(AUROC = %0.2f)' %rSVR_auc)

# plt.title('ROC plot')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()