import numpy as np
from numpy import genfromtxt
import networkx as nx
import matplotlib.pyplot as plt


dt = [('len', float)]

labels = ['aggressive','caring','confident','dominant','emotionallystable','intelligent','mean','responsible','sociable','trustworthy','unhappy','weird']
my_data = genfromtxt('TLr1_subjectiveDM.csv', delimiter=',', skip_header=1)
my_data = np.delete(my_data, 0, 1)
my_data = my_data.view(dt)

G = nx.from_numpy_matrix(my_data)



G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),labels)))

G = nx.drawing.nx_agraph.to_agraph(G)

G.node_attr.update(color="red", style="filled")
G.edge_attr.update(color="blue", width="2.0")

G.draw('out.png', format='png', prog='neato')
