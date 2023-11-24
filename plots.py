# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:29:03 2023

@author: rhanusa
"""

import numpy as np
import matplotlib.pyplot as plt

from network_model import x_values

coordinates = np.genfromtxt('coords.csv', delimiter=',')

# routes = []

# for bike in range(len(x_values[0,0,:])):
#     route = []
#     for i in range(len(x_values[:,0,0])):
#         for j in range(len(x_values[0,:,0])):
#             if x_values[i, j, bike] < 1:
#                 pass
#             else:
#                 j = i-1
#                 route.append(coordinates[i])
#     routes.append(route)
            

routes = []

# for bike in range(len(x_values[0,0,:])):
#     route = []
#     for i in range(len(x_values[:,0,0])):
#         for j in range(len(x_values[0,:,0])):
#             if x_values[i, j, bike] < 1:
#                 print('i: ', i, ', j: ', j)
#                 pass
#             else:
#                 print(f'bike: {bike}, j: {j}, i: {i}')
#                 route.append(coordinates[j])
#                 i = j 
#                 j = 0
#                 pass
#     routes.append(route)
    



node_routes = []

for bike in range(len(x_values[0,0,:])):
    node_list = []
    i = 0
    flag = True
    while i < len(x_values[:,0,0])-1 and i not in node_list:
        
        for j in range(len(x_values[0,:,0])):
            print(f'True, i={i}, j={j}')
            if x_values[i,j,bike] == 1:
                flag = False
                print(f'True, i={i}, j={j}')
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



route = []
for i in node_list:
    route.append(coordinates[i])
    
# Make route circular by adding back first element to end of list.
route.append(route[0])

    
            
    
    
bike1 = x_values[:,:,0]
bike2 = x_values[:,:,1]
bike3 = x_values[:,:,2]
bike4 = x_values[:,:,3]
bike5 = x_values[:,:,4]
bike6 = x_values[:,:,5]


# Plot the coordinates
plt.scatter(coordinates[:, 1], coordinates[:, 0])
plt.title('Pharmacies')
plt.show()