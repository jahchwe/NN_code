# import pandas as pd
import numpy as np
from numpy import genfromtxt
import networkx as nx
# import matplotlib.pyplot as plt
# import pygraphviz as pg

labels = ['aggressive','caring','confident','dominant','emotionally stable','intelligent','mean','responsible','sociable','trustworthy','unhappy','weird']
my_data = genfromtxt('TLr1_subjectiveDM.csv', delimiter=',', skip_header=1)
my_data = np.delete(my_data, 0, 1)

# get sum of all values for later
sum=0;
for row in range (len(my_data)):
    for col in range(len(my_data[0])):
      sum += my_data[row][col]

sum /= 15

# dt = [('len',  float)]
# create values for length of nodes and pen width
dt = np.dtype({'names': ['len', 'penwidth'], 'formats':[float, float]})
my_data = np.array(my_data, dt)

# now reassign the values of penwidth inversely proportional to length values

for row in range (len(my_data)):
    for col in range(len(my_data[0])):
      new_val = np.divide(sum,my_data[row][col][1])
      my_data[row][col][1] = new_val
print(my_data)

# G = nx.from_numpy_matrix(my_data)

# G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),labels)))

# G = nx.drawing.nx_agraph.to_agraph(G)

# G.write('dotfiles/dotGenerated.dot')
# G.append('dotfiles/dotGenerated.dot')

# G.node_attr.update(color="black")
# G.node_attr.update(style="filled")
# G.edge_attr.update(color="gray", width="0.3")


# G.draw('graph.png', format='png', prog='neato')

# B = pg.AGraph('simple.dot') # create a new graph from file
# B.layout() # layout with default (neato)
# B.draw('simple.png')
