import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('data_python.csv')
P2 = 1 - data['P1']
data['P2'] = P2
label = ['$u_{i}$', '$u_{j}$', '$dp_{i}$', '$dp_{j}$', '$y_{i}$', '$y_{j}$', '$r_{i}$', '$r_{j}$', '$p_{i}$', '$p_{j}$', '$x_{i}$']
f_list = ['U1', 'U2', 'DP1', 'DP2', 'Y1', 'Y2', 'R1', 'R2', 'P1', 'P2', 'X1']
df = data[f_list]
dfData = df.corr()
f, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(dfData, annot=True, square=False, cmap="Oranges")
ax.set_xticklabels(label, fontsize=12)
ax.set_yticklabels(label, fontsize=12)
f.show()
plt.show()
