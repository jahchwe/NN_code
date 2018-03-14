import pandas as pd
import numpy as np
import networkx as nx
from numpy import genfromtxt
from sklearn import cluster
from sklearn.cluster import KMeans

# read from file
labels = ['aggressive','caring','confident','dominant','egotistic','emotionally stable','intelligent','mean','responsible','sociable','trustworthy','unhappy','weird']

data = genfromtxt('data/conceptSM.csv', delimiter=',', skip_header=1)
my_data = np.delete(data, 0, 1)

dt = [('len',  float)]

# create values for length of nodes and pen width
# dt = np.dtype({'names': ['len', 'weight'], 'formats':[float, float]})
print(my_data)

for row in range (len(my_data)):
    for col in range(len(my_data[0])):
      my_data[row][col]+=1
my_data = np.array(my_data, dt)

G = nx.from_numpy_matrix(my_data)
G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),labels)))
G = nx.drawing.nx_agraph.to_agraph(G)

# generate dotfile
G.write('dotfiles/conceptSM.dot')

estimator = KMeans(n_clusters=3)
estimator.fit(my_data)
cluster_grouping = {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}

named_clusters = [];
for x in range (len(cluster_grouping)):
  for y in cluster_grouping[x]:
    named_clusters.append(labels[y])

print(named_clusters)

# TODO: write directly to dotfile
# - add xlabel
# - color node by cluster
# - remove self-referencing edges
# - color edges between cluster ndoes