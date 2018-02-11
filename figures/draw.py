import pygraphviz as pg
B = pg.AGraph('dotfiles/dotGenerated.dot') # create a new graph from file
B.layout() # layout with default (neato)
B.draw('img/test.png')