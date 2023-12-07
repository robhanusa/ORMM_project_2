# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:29:03 2023

@author: rhanusa
"""

import numpy as np
import matplotlib.pyplot as plt

# from network_model import x_values

# coordinates = np.genfromtxt('coords.csv', delimiter=',')          

x_values = np.load('x_values4.npy')

# Create node map to allow us to put satellites at beginning or end
nmap = {
    0: 7,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 13,
    13: 14,
    14: 15,
    15: 12,
    16: 7,
    17: 7,
    18: 7,
    19: 12,
    20: 12,
    21: 12
    }
    
def make_routes_per_sat(x_values, sat, satellites):    
    node_routes = []
    i = 0
    flag = True
    for i in range(len(x_values[:,0])):
        if x_values[i,sat] == 1:
            node_list = [sat, i]
            node_list_map = [nmap[sat], nmap[i]]
            i_route = i
            c = 0
            while i_route < len(x_values[:,0]) and all(sat not in node_list[1:] for sat in satellites) and c < 100: 
                for j in range(len(x_values[0,:])):
                    if x_values[j, i_route] == 1:
                        flag = False
                        node_list_map.append(nmap[j])
                        node_list.append(j)
                        i_route = j
                        break
                    else:
                        flag = True
                        pass
                c += 1
                if flag:
                    i_route += 1
                    flag = False
            
            #
            node_routes.append(node_list_map)
        
    return node_routes


def make_node_routes(x_values, satellites):
    node_routes = []
    for sat in satellites:
        routes = make_routes_per_sat(x_values, sat, satellites)
        for route in routes:
            node_routes.append(route)
        
    return node_routes
    

customer_routes = make_node_routes(x_values, (0,15))


def make_coordinate_routes(node_routes, coordinates):
    routes = []
    
    for node_list in node_routes:
        route = []
        for i in node_list:
            route.append(coordinates[i])
        
        routes.append(route)
        
    return routes


#%%

def plot_routes(x_values, satellites):
    
    coordinates = np.genfromtxt('coords.csv', delimiter=',')
    
    node_routes = make_node_routes(x_values, satellites)
    
    routes = make_coordinate_routes(node_routes, coordinates)
    
    fig, ax = plt.subplots()
    
    for route in routes:
        x = [route[i][1] for i in range(len(route))]
        y = [route[i][0] for i in range(len(route))]
        plt.plot(x,y) #, color='b')
    
    # Plot the coordinates
    ax.scatter(coordinates[:, 1], coordinates[:, 0])
    for i, coord in enumerate(coordinates):
        
        # Uncomment below to label points according to prompt
        # ax.annotate(str(i+1), (coord[1], coord[0]), xytext = (5,0), 
        #             textcoords='offset pixels')
        ax.annotate(str(i), (coord[1], coord[0]), xytext = (5,0), 
                    textcoords='offset pixels')
    
    
    plt.title('Pharmacies')


#%%

# import pandas as pd

# t_values = np.load('t_values.npy')

# bike_time_order = [index for index, value in sorted(enumerate(t_values), key=lambda x: x[1])]
# bike_times = [t_values[i] for i in bike_time_order]
# mapped_order = [nmap[i] for i in bike_time_order]

# df = pd.DataFrame({'customer': mapped_order,'time': bike_times, 'node': bike_time_order})

#%% How long does each bike route take?

tau = np.genfromtxt('time_matrix_bike.csv', delimiter=',')  

times = []
for route in customer_routes:
    time = 0
    for stop in range(len(route)-1):
        time += tau[route[stop], route[stop+1]]
    times.append(time)
    
