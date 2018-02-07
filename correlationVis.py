import pandas as pd
import numpy as np
from numpy import genfromtxt
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
from sklearn import cluster



labels = ['aggressive','caring','confident','dominant','emotionally stable','intelligent','mean','responsible','sociable','trustworthy','unhappy','weird']
data = genfromtxt('TLr1_subjectiveDM.csv', delimiter=',', skip_header=1)
my_data = np.delete(data, 0, 1)

# for row in range (len(my_data)):
#     for col in range(len(my_data[0])):
#       my_data[row][col] *= 55


dt = [('len',  float)]
# create values for length of nodes and pen width

# dt = np.dtype({'names': ['len', 'weight'], 'formats':[float, float]})
my_data = np.array(my_data, dt)

G = nx.from_numpy_matrix(my_data)

G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),labels)))

# print(nx.clustering(G, None, weight='weight'))

G = nx.drawing.nx_agraph.to_agraph(G)
# generate dotfile
# G.write('dotfiles/dotGenerated.dot')

estimator = KMeans(n_clusters=3)
estimator.fit(my_data)

cluster_grouping = {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}

print(cluster_grouping)



# # get sum of all values for later
# sum=0;

# for row in range (len(my_data)):
#     for col in range(len(my_data[0])):
#       sum += my_data[row][col]


# # for row in range (len(my_data)):
# #     for col in range(len(my_data[0])):
# #       new_val = np.divide(np.divide(sum,my_data[row][col][1]),30)
# #       my_data[row][col][1] = new_val
