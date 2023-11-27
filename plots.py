# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:29:03 2023

@author: rhanusa
"""

import numpy as np
import matplotlib.pyplot as plt

#from network_model import x_values

# coordinates = np.genfromtxt('coords.csv', delimiter=',')          

# x_values = np.load('x_values3.npy')

# bike1 = x_values[:,:,0]
# bike2 = x_values[:,:,1]
# bike3 = x_values[:,:,2]
# bike4 = x_values[:,:,3]
# bike5 = x_values[:,:,4]
# bike6 = x_values[:,:,5]

def make_node_routes(x_values):
    node_routes = []
    
    for bike in range(len(x_values[0,0,:])):
        node_list = []
        i = 0
        flag = True
        while i < len(x_values[:,0,0]) and i not in node_list:
            for j in range(len(x_values[0,:,0])):
                if x_values[i,j,bike] == 1:
                    flag = False
                    node_list.append(i)
                    i=j
                    break
                else:
                    flag = True
                    pass
                
            if flag:
                i += 1
                flag = False
        
        node_routes.append(node_list)
        
    return node_routes

# node_routes = make_node_routes(x_values)



def make_coordinate_routes(node_routes, coordinates):
    routes = []
    
    for node_list in node_routes:
        route = []
        for i in node_list:
            route.append(coordinates[i])
        
        # Make route circular by adding back first element to end of list.
        route.append(route[0])
        
        routes.append(route)
        
    return routes


#%%

def plot_routes(x_values):
    
    coordinates = np.genfromtxt('coords.csv', delimiter=',')
    
    node_routes = make_node_routes(x_values)
    
    routes = make_coordinate_routes(node_routes, coordinates)
    
    fig, ax = plt.subplots()
    
    for route in routes:
        x = [route[i][1] for i in range(len(route))]
        y = [route[i][0] for i in range(len(route))]
        plt.plot(x,y)
    
    # Plot the coordinates
    ax.scatter(coordinates[:, 1], coordinates[:, 0])
    for i, coord in enumerate(coordinates):
        ax.annotate(str(i), (coord[1], coord[0]), xytext = (5,0), 
                    textcoords='offset pixels')
    
    
    plt.title('Pharmacies')
