import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objs as go
import plotly
plotly.offline.iplot
from plotly import subplots



def calcul_periode_obs(d):
    obs = d.groupby(['collectivite'])['year_event'].min().rename("obs_start").reset_index(drop=False)
    d = pd.merge(d, obs, how='left', on='collectivite')
    obs = d.groupby(['collectivite'])['year_event'].max().rename("obs_end").reset_index(drop=False)
    d = pd.merge(d, obs, how='left', on='collectivite')
    return d

# Path to data
PATH = "../data/"

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

# Encoding quantitative variables : oneHotEncoding On supprime MATERIAU car l'info est contenue dans MATAGE
# 1 - on supprime les matages qu'on peut pas utiliser pour le onehotencoding
df_T = df_T.drop(df_T[df_T.MATAGE == 'ACIER19201940'].index)
df_T = df_T.drop(df_T[df_T.MATAGE == 'BONNA19651975'].index)
df_T = df_T.drop(df_T[df_T.MATAGE == 'FONTEDUCTILE20102020'].index)


df_T = pd.concat([df_T, pd.get_dummies(df_T.collectivite)], axis = 1)
df_T = pd.concat([df_T, pd.get_dummies(df_T.MATAGE)], axis = 1)

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
# On supprime les colonnes intermédiaires, matériau et les tuyaux qui on une durée de vie > 63 ans et = 0 ans
df_casse = df_casse.drop(columns=['derniere_casse','ID2', 'DDCC2'])
df_casse = df_casse.drop(df_casse[df_casse.duree_de_vie > 63].index)
#df_test = df_casse.groupby(df_casse.MATAGE).size()
df_casse = df_casse.drop(df_casse[df_casse.duree_de_vie < 5].index)


# Standardizing quantitative data
scaler = StandardScaler()
df_casse['DIAMETRE_std'] = scaler.fit_transform(df_casse[['DIAMETRE']])
df_casse['year_pose_std'] = scaler.fit_transform(df_casse[['year_pose']])
df_casse = df_casse.drop(columns=['DIAMETRE','IDT'])

# # Encoding quanlitative variables : oneHotEncoding On supprime MATERIAU car l'info est contenue dans MATAGE
# df_casse = pd.concat([df_casse, pd.get_dummies(df_casse.collectivite)], axis = 1)
# df_casse = pd.concat([df_casse, pd.get_dummies(df_casse.MATAGE)], axis = 1)

#splittting Data
p_train = 0.70
Y = df_casse.duree_de_vie
X = df_casse.drop(columns=['ID', 'limit_low', 'DDCC', 'obs_start', 'obs_end', 'DDP', 'year_event', 'collectivite', 'MATERIAU', 'duree_de_vie', 'year_pose'])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1-p_train, shuffle = True, stratify = X.MATAGE, random_state=0)
list_mat = list(x_test['MATAGE'])
list_mat1 =sorted( list(x_test['MATAGE'].unique()))

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
df_y_pred_test = pd.DataFrame(y_pred).rename(columns = {0: "dure_vie_pred"})
df_y_pred_test['MATAGE'] = list_mat
df_y_pred_test['dure_vie_test'] = list(y_test)
# fonction pour les couleurs
clrs = []
for mat in df_y_pred_test['MATAGE']:
    if mat == "FONTEDUCTILE19661980":
        clrs.append("rgb(90, 44, 158, 1.0)")
    elif mat == "FONTEGRISE19401970":
        clrs.append("rgb(128, 128, 128, 1.0)")
    elif mat == "PVC19902020":
        clrs.append("rgb(0, 0, 0, 1.0)")
    elif mat == "FONTEDUCTILE20002010":
        clrs.append("rgb(42, 227, 210, 1.0)")
    elif mat =="FONTEDUCTILE19801990":
        clrs.append( "rgb(42, 227, 39, 1.0)")
    elif mat == "PVC19701990":
        clrs.append("rgb(223, 227, 39, 1.0)")
    elif mat == "FONTEDUCTILE19511966":
        clrs.append("rgb(223, 0, 0, 1.0)")
    elif mat == "FONTEDUCTILE19902000":
        clrs.append("rgb(211, 192, 233, 1.0)")
    elif mat == "PEHD19581995":
        clrs.append("rgb(211, 78, 233, 1.0)")
    elif mat == "FONTEGRISE18001900":
        clrs.append("rgb(29, 78, 233, 1.0)")
    elif mat == "PVC19481970":
        clrs.append("rgb(29, 78, 0, 1.0)")
    elif mat == "PEHD19952006":
        clrs.append("rgb(29, 78, 124, 1.0)")
    elif mat == "BONNA18001965":
        clrs.append("rgb(29, 255, 255, 1.0)")
    elif mat == "BONNA19752020":
        clrs.append("rgb(101, 211, 0, 1.0)")
    elif mat == "FONTEGRISE19001930":
        clrs.append("rgb(209, 211, 0, 1.0)")
    elif mat == "ACIER19802020":
        clrs.append("rgb(255, 0, 142, 1.0)")
    elif mat == "FONTEDUCTILE20102020":
        clrs.append("rgb(255, 139, 142, 1.0)")
    elif mat == "PEBD19552020":
        clrs.append("rgb(255, 139, 34, 1.0)")
    elif mat == "FONTEGRISE19301940":
        clrs.append("rgb(255, 255, 34, 1.0)")
    elif mat == "PEHD20062020":
        clrs.append("rgb(106, 0, 212, 1.0)")
    else:
        clrs.append("rgb(106, 0, 30, 1.0)")
        
df_y_pred_test['Couleur'] = clrs

data = []
for mat in list_mat1:
    print(mat)
    print(list(df_y_pred_test[df_y_pred_test.MATAGE == mat].dure_vie_pred))
    print(list(df_y_pred_test[df_y_pred_test.MATAGE == mat].dure_vie_test))
    trace1 = go.Scatter(
        name = mat,
        x = list(df_y_pred_test[df_y_pred_test.MATAGE == mat].dure_vie_pred),
        y = list(df_y_pred_test[df_y_pred_test.MATAGE == mat].dure_vie_test),
        mode = "markers",
        marker = dict(color = df_y_pred_test[df_y_pred_test.MATAGE == mat].Couleur),
        text =mat
        )
    data.append(trace1)
trace2 = go.Scatter(
                x = [0, 61],
                y = [0, 61],
                mode = "lines",
                name = '[0, 61]')
data.append(trace2)
    
layout = go.Layout(title="Observé en fonction du prédit", yaxis=dict(title = "Observé"),
                   xaxis=dict(title='Prédit'),
                   )

fig = go.Figure(data = data, layout = layout)
fig.show()  

#### Box plot pour les résultats
nb_tuyaux = df_y_pred_test[['MATAGE', 'dure_vie_pred']].groupby(['MATAGE']).size()

fig = subplots.make_subplots(rows=2, cols=1,
                                 shared_xaxes=True,
                                 shared_yaxes=True, vertical_spacing=0.01,
                                 row_width=[0.8, 0.2],
                                 )
data = []
for mat in list_mat1:
    trace1 = go.Box(
        y = list(df_y_pred_test[df_y_pred_test.MATAGE == mat].dure_vie_pred),
        name =  mat,
        showlegend=False
        )
    fig.add_trace(trace1, row=2, col=1)
    #data.append(trace1)
# fig.update_layout(yaxis=dict(title = "Durée de vie prédit"),
#                    xaxis=dict(title='MATAGE'))
# Histogramme X
trace2 = go.Bar(
    x = list_mat1,
    y = list(nb_tuyaux),
    name = 'Nombre de casses par MATAGE',
    marker = dict(color='rgba(0, 0, 150, 0.6)', line=dict(color='rgba(0, 0, 150, 0.6)', width=1)),
    text = list(nb_tuyaux), textposition="auto",
    showlegend=False
)
# Top left
fig.add_trace(trace2, row=1, col=1)

fig.update_layout(title_text="Box plot pour la durée de vie de chaque MATAGE", 
                  yaxis=dict(title = "Nombre de casses"))    
fig.show()  

######### estimer une durée de vie et une date de casse pour les tuyaux sélectionnés
# sélectionner que les tuyaux qui n'ont pas une date de casse
df_Non_casse = df_T.loc[~df_T.index.isin(df_T.dropna().index)]
df_Non_casse = df_Non_casse.drop(df_Non_casse[df_Non_casse.MATAGE == 'ACIER19201940'].index)
df_Non_casse = df_Non_casse.drop(df_Non_casse[df_Non_casse.MATAGE == 'BONNA19651975'].index)

# Standardizing quantitative data
df_Non_casse['DIAMETRE_std'] = scaler.fit_transform(df_Non_casse[['DIAMETRE']])
df_Non_casse['year_pose_std'] = scaler.fit_transform(df_Non_casse[['year_pose']])
df_Non_casse = df_Non_casse.drop(columns=['IDT', 'DIAMETRE'])
list_mat_non_casse = list(df_Non_casse['MATAGE'].unique())
list_mat_non_casse1 = list(df_Non_casse['MATAGE'])
#supprimer les colonnes inutiles
df_Non_casse = df_Non_casse.drop(columns=['MATAGE', 'ID', 'limit_low', 'DDCC', 'obs_start', 'obs_end', 'DDP', 'year_event', 'collectivite', 'MATERIAU', 'year_pose'])

#appliquer le modèle de regression
y_pred_non_casse = regressor.predict(df_Non_casse)

############################################ visualisation

########## Courbe nombre de casse en fonction du matage en utilisant df_casse 
    
list_mat_df_casse = list(df_casse['MATAGE'])
list_mat_df_casse1 =sorted( list(df_casse['MATAGE'].unique()))

nb_tuyaux_casse_df_casse = df_casse.groupby(['MATAGE']).size()
data = []

# Histogramme X
trace4 = go.Bar(
    x = list_mat_df_casse1,
    y = list(nb_tuyaux_casse_df_casse),
    name = 'Nombre de casses par MATAGE',
    marker = dict(color='rgba(0, 0, 150, 0.6)', line=dict(color='rgba(0, 0, 150, 0.6)', width=1)),
    text = list(nb_tuyaux_casse_df_casse), textposition="auto",
    showlegend=False
)
data.append(trace4)

fig = go.Figure(data = data)

fig.show() 















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