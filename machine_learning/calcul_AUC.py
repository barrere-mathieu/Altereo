import pandas as pd
from sklearn.metrics import auc
from matplotlib import pyplot as plt

def plot_AUC(df, annotations):
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

    plt.show()
    return auc_measure


def save_AUC(df, file_name, annotations):
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
    plt.savefig(str(file_name) + '.png')
    plt.show()
    return auc_measure


# Test
# df = pd.DataFrame(data={'event': [0, 1, 1, 1],
#                         'proba': [0.3, 0.75, 0.77, 0.88]})
# annotations = [('Model', 'Random Survival Forest'), ('Année prédiction', 2008), ('Durée Prédiction (année)', 2), ('Nombre de collectivité', 13), ('Nombre tuyaux total', 463793), ('Nombre de casse', 865)]
#
# save_AUC(df, 'test fig', annotations)
