import pandas as pd
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sksurv.metrics import cumulative_dynamic_auc
import numpy as np


# def plot_AUC(df, annotations, display = False, save = True):
#     # Structure du DF : une colonne proba avec la probabilité de casse et une colonne event (0/1) décrivant si un tuyau
#     # est cassé
#     df2 = df[['proba', 'event']]
#     df2 = df2.sort_values(['proba'], ascending=[False])
#     df2['ranks'] = df2['proba'].rank(ascending=False)
#     df2['ranks_pct'] = df2['ranks'] / max(df2['ranks'])
#     df2['casses_cumul'] = df2.event.cumsum()
#     df2['casse_pct'] = df2['casses_cumul'] / sum(df2.event)
#     x = pd.concat([pd.Series([0]), df2.ranks_pct])
#     y = pd.concat([pd.Series([0]), df2.casse_pct])
#     auc_measure = round(auc(df2.ranks_pct, df2.casse_pct), 2)
#     annotations.append(('AUC', auc_measure))
#
#     plt.plot(x, y, color = 'b', label='Prediction')
#     plt.plot([0, 1], [0, 1], color='r', label = 'Ref 50%')
#     plt.legend(loc='best')
#     plt.ylabel('% de survie modèle')
#     plt.xlabel('% de casse cummulé réel')
#     plt.title('ROC casses')
#
#     txt = ''
#     for e in annotations:
#         # annotation structure: liste de (Txt, Valeur)
#         txt += e[0] + ' : ' + str(e[1])+'\n'+'\n'
#     plt.xlim(0, 1)
#     plt.ylim(0, 1.1)
#     plt.text(-1.2, 0.3, txt, fontsize=10)
#     plt.grid(True)
#     plt.subplots_adjust(left=0.5)
#
#     plt.show()
#     return auc_measure


def save_AUC(df, file_name, annotations, display = False, save = True):
    # Structure du DF : une colonne proba avec la probabilité de casse et une colonne event (0/1) décrivant si un tuyau
    # est cassé
    df2 = df[['proba', 'event']]
    df2 = df2.sort_values(['proba'], ascending=[False])
    df2['ranks'] = df2['proba'].rank(ascending=False)
    df2['ranks_pct'] = df2['ranks'] / max(df2['ranks'])
    df2['casses_cumul'] = df2.event.cumsum()
    df2['casse_pct'] = df2['casses_cumul'] / sum(df2.event)
    x = pd.concat([pd.Series([0]), df2.ranks_pct])
    y = pd.concat([pd.Series([0]), df2.casse_pct])
    auc_measure = round(auc(df2.ranks_pct, df2.casse_pct), 2)
    annotations.append(('AUC', auc_measure))

    plt.figure()
    plt.plot(x, y, color = 'b', label='Prediction')
    plt.plot([0, 1], [0, 1], color='r', label = 'Ref 50%')
    plt.legend(loc='best')
    plt.ylabel('% de survie modèle')
    plt.xlabel('% de casse cummulé réel')
    plt.title('ROC casses')

    txt = ''
    for e in annotations:
        # annotation structure: liste de (Txt, Valeur)
        txt += e[0] + ' : ' + str(e[1])+'\n'+'\n'
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    plt.text(-1.2, 0.3, txt, fontsize=10)
    plt.grid(True)
    plt.subplots_adjust(left=0.5)
    if save:
        plt.savefig(str(file_name) + 'AUC.png')
    if display:
        plt.show()
    return auc_measure

def stardard_ROC(df, file_name, display = False, save = True):
    y_test = df.event
    y_score = df.proba
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    if save:
        plt.savefig(str(file_name) + '_ROC.png')
    if display:
        plt.show()


def plot_cumulative_dynamic_auc_auto(risk_score, times, y_train, y_test, model_name, label='', color=None, display = False, save = True):
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_score, times)

    plt.figure()
    plt.plot(times, auc, marker="o", color=color, label=label)
    plt.xlabel("days from enrollment")
    plt.ylabel("time-dependent AUC")
    plt.axhline(mean_auc, color=color, linestyle="--")
    plt.legend()
    if display:
        plt.show()
    if save:
        plt.savefig(str(model_name) + '_AUTO_dynamicAUC.png')

def plot_cumulative_dynamic_auc(df, times, model_name, display = False, save = True):
    AUCs = []
    # Calcul AUC globale
    y_test = df.event
    y_score = df.proba
    fpr, tpr, _ = roc_curve(y_test, y_score)
    mean_auc = auc(fpr, tpr)

    # Calcul des AUC time dependent
    for t in times:
        df['event_time'] = 0
        df.loc[(df.duree_de_vie <= t) & (df.event == 1), 'event_time'] = 1
        y_test = df.event_time
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        AUCs.append(roc_auc)
        if save:
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic curve with DDV <= '+str(t))
            plt.legend(loc="lower right")
            plt.savefig(str(model_name) + "_" + str(t) +'_ROC.png')

    plt.figure()
    plt.xlabel("years from observation windows starts")
    plt.ylabel("time-dependent AUC")
    plt.axhline(mean_auc, color='navy', linestyle="--", label = 'mean_AUC')
    plt.legend()
    plt.plot(times, AUCs, marker="o", color = 'darkorange', label=model_name)
    if display:
        plt.show()
    if save:
        plt.savefig(str(model_name) + '_dynamicAUC.png')

    return mean_auc, np.mean(AUCs), AUCs



# # Test
# df = pd.DataFrame(data={'event': [0, 1, 1, 1],
#                         'proba': [0.3, 0.75, 0.77, 0.88]})
# annotations = [('Model', 'Random Survival Forest'), ('Année prédiction', 2008), ('Durée Prédiction (année)', 2), ('Nombre de collectivité', 13), ('Nombre tuyaux total', 463793), ('Nombre de casse', 865)]
#
# # save_AUC(df, 'test fig', annotations)
# stardard_ROC(df)