import pandas as pd
from sklearn.metrics import auc
from matplotlib import pyplot as plt

def AUC(df):
    # Structure du DF : une colonne proba avec la probabilité de casse et une colonne event (0/1) décrivant si un tuyau
    # est cassé
    df2 = df[['proba', 'event']]
    df2 = df2.sort_values(['proba'], ascending=[False])
    df2['ranks'] = df2['proba'].rank(ascending=False)
    df2['ranks_pct'] = df2['ranks'] / max(df2['ranks'])
    df2['casses_cumul'] = df2.event.cumsum()
    df2['casse_pct'] = df2['casses_cumul'] / sum(df2.event)

    plt.plot(df2.ranks_pct, df2.casse_pct, '-ok')
    plt.show()
    auc_measure = round(auc(df2.ranks_pct, df2.casse_pct), 2)
    print(auc_measure)
    return auc_measure

# Test
df = pd.DataFrame(data={'event': [0, 1, 1, 1],
                        'proba': [0.3, 0.75, 0.77, 0.88]})
AUC(df)
