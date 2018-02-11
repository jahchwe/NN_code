import pandas as pd
import numpy as np
import networkx as nx
from numpy import genfromtxt
from sklearn import cluster
from sklearn.cluster import KMeans

# read from file
labels = ['aggressive','caring','confident','dominant','emotionally stable','intelligent','mean','responsible','sociable','trustworthy','unhappy','weird']
data = genfromtxt('TLr1_subjectiveDM.csv', delimiter=',', skip_header=1)
my_data = np.delete(data, 0, 1)

dt = [('len',  float)]
# create values for length of nodes and pen width

# dt = np.dtype({'names': ['len', 'weight'], 'formats':[float, float]})
my_data = np.array(my_data, dt)

G = nx.from_numpy_matrix(my_data)
G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),labels)))
G = nx.drawing.nx_agraph.to_agraph(G)

# generate dotfile
G.write('dotfiles/dotGenerated.dot')

estimator = KMeans(n_clusters=3)
estimator.fit(my_data)
cluster_grouping = {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}

print(cluster_grouping)