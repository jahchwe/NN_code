#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:45:10 2017

@author: jahchwe

Animating a neural network using matplotlib. 

Neuron code and generation taken from:
    https://github.com/miloharper/visualise-neural-network

Animation created with help from matplotlib documentation and:
    http://firsttimeprogrammer.blogspot.com/2014/12/animated-graphs-with-matplotlib.html
    https://nickcharlton.net/posts/drawing-animating-shapes-matplotlib.html
    
Strategy: 
    1. use NN functions to create node locations and connections. 
    2. Store this info (cannot store actual figure objects)
    3. Recreate graph at each animation iteration using stored info, adjusting color to reflect activation
"""

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np 
import matplotlib.animation as animation

#activation_data = np.genfromtxt("
 
#circle center info 
counter = 0
xs = []
ys = []

#line info to be redrawn
point1x = []
point1y = []
point2x = []
point2y = []

class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        global counter 
        global xs,ys
        #use global counter to keep track of node position
        
        #calculate color using numpy array of activation values
        #color = pyplot.cm.plasma(activations[0[1]*250 * scalar])
        #use iterators to keep track of which node in layer. 
        #if layer_counter > static preknown activation layer size, layer_counter = 0
        color = pyplot.cm.plasma(counter*20)
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fc = color, ec = "black", fill=True)
        xs.append(self.x)
        ys.append(self.y)
        pyplot.gca().add_patch(circle)
        '''
        Trying to collect circles to change color
        !!!after circle has been added, cannot change color
        Store copies of lines and circles to edit to recreate at each iteration
        '''
        counter += 1
        return circle


class Layer():
    nodes = []
    def __init__(self, network, number_of_neurons):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        
        #my addition. Store line info
        
        global point1x, point1y, point2x, point2y
        
        #
        
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), color = "black")
        
        #look at line2D constructor
        point1x.append(neuron1.x - x_adjustment)
        point2x.append(neuron2.x + x_adjustment)
        point1y.append(neuron1.y - y_adjustment)
        point2y.append(neuron2.y + y_adjustment)
        
        pyplot.gca().add_line(line)

    def draw(self):
        for neuron in self.neurons:
            '''my addition is temp'''
            temp = neuron.draw()
            self.nodes.append(temp)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)


class NeuralNetwork():
    nodes = []
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons):
        layer = Layer(self, number_of_neurons)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        self.getNodes()
        for q in network.nodes:
            q.set_fc = "black"
        pyplot.axis('scaled')
        pyplot.savefig("exampleNN.png", format="png", dpi = 900)
        pyplot.show()
        
    def getNodes(self):
        for layer in self.layers:
            for n in layer.nodes:
                self.nodes.append(n)
                
'''
if __name__ == "__main__":
    vertical_distance_between_layers = 6
    horizontal_distance_between_neurons = 2
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 8
    network = NeuralNetwork()
    network.add_layer(1)
    network.add_layer(6)
    #network.add_layer(1)
    network.draw()
    """
    nodes only exist after drawing
    """
    network.getNodes()
    
    
    pyplot.show()
'''
#specify network dimensions

vertical_distance_between_layers = 6
horizontal_distance_between_neurons = 2
neuron_radius = 0.5
number_of_neurons_in_widest_layer = 6
network = NeuralNetwork()
network.add_layer(1)
network.add_layer(6)

#make network, this process will also save the graph details
network.draw()

'''

activation_levels = np.genfromtxt("singleton_output.csv", delimiter = ",")

#create new nodes for each animation slide 
#my additions


figure = pyplot.figure()
    
circles = []

counter = 0

def init():
    global circles
    circle = pyplot.Circle((xs[0],ys[0]), radius=0.5, fc = "black", ec = "black", fill=True)
    figure.gca().add_patch(circle)
    for i in range(1,len(xs)):
        color_val = activation_levels[0][i-1]
        if (color_val < 0): 
            color_val = 0
        circle = pyplot.Circle((xs[i], ys[i]), radius=0.5, fc = pyplot.cm.plasma(color_val * 500), ec = "black", fill=True)
        circles.append(circle)
        figure.gca().add_patch(circle)
    for q in range(len(point1x)):
        line = pyplot.Line2D((point1x[q], point2x[q]), (point1y[q], point2y[q]), color = "black")
        figure.gca().add_line(line)
    
    
    pyplot.axis("scaled")
    return circles
        
def animate(j):
    global circles
    global counter

    for i in range(1,len(xs)):
        if (counter > 5):
            counter = 0
        color_val = activation_levels[j][counter] * 2
        
        if (color_val < 0): 
            color_val = 0
        circles[i-1].set_fc(pyplot.cm.plasma(color_val))
        #print(color_val*250)
        counter+=1
        
        figure.gca().add_patch(circles[i-1])
    return circles


anim=animation.FuncAnimation(figure,animate,init_func=init,frames=50,blit=True)

anim.save('big_animation2.mp4', fps=2, dpi = 200, extra_args=['-vcodec', 'libx264'])
    
'''    
#init()

pyplot.axis("scaled")
pyplot.show()


