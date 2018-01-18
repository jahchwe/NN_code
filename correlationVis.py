# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt

import csv

trait_correlations = []
r = csv.reader(open('TLr1_subjectiveDM.csv', newline=''), delimiter=',')
for row in r:
  trait_correlations.append(row)

# get all the traits
nodes = trait_correlations[0]
nodes = nodes[1:len(nodes) -1]

print(nodes)

# G = nx.Graph()
# G.add_nodes_from(traits)
# # (2, 3, {'weight': 3.1415})

# nx.draw(G, with_labels="true")
# plt.savefig("test.png") # save as png
# plt.show() # display