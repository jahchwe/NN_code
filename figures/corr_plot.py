import numpy as np
from numpy import genfromtxt
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

labels = ['aggressive','caring','confident','dominant','emotionally stable','intelligent','mean','responsible','sociable','trustworthy','unhappy','weird']
data = genfromtxt('TLr1_subjectiveDM.csv', delimiter=',', skip_header=1)
data = np.delete(data, 0, 1)

cmap = sns.diverging_palette(220, 10, as_cmap=True)

cg = sns.clustermap(data, center=0, cmap=cmap)
cg.ax_row_dendrogram.set_visible(False)
cg.ax_col_dendrogram.set_visible(False)
plt.show()

