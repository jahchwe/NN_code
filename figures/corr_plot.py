import numpy as np
from numpy import genfromtxt
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

labels = ['aggressive','caring','confident','dominant','emotionally stable','intelligent','mean','responsible','sociable','trustworthy','unhappy','weird']
data = genfromtxt('TLr1_subjectiveDM.csv', delimiter=',', skip_header=1)
data = np.delete(data, 0, 1)

estimator = KMeans(n_clusters=3)
estimator.fit(data)
cluster_grouping = {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}

# create the reordered indices based on clustering
new_order = []
for i in cluster_grouping:
  for n in cluster_grouping[i].tolist():
    new_order.append(n)

reordered_labels = []
for i in new_order:
  reordered_labels.append(labels[i])

# reorder the rows based on clustering
clustered_corrs = []
for i in new_order:
  clustered_corrs.append(data[i])

# reorder columns based on clustering
count = 0;
for row in clustered_corrs:
  new_row = [];
  for col in new_order:
    new_row.append(row[col])
  clustered_corrs[count] = new_row
  count = count + 1;

print(clustered_corrs)
cmap = sns.diverging_palette(220, 10, as_cmap=True)

fig, axs = plt.subplots(figsize=(11, 9))

sns.heatmap(clustered_corrs, xticklabels = reordered_labels, yticklabels = reordered_labels, cmap=cmap)

plt.xticks(rotation=45)

plt.show()

