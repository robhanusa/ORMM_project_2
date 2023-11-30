# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:37:07 2023

@author: rhanusa
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

import plots as pl
from plots import node_map

# ---- Global variables ----
M = 1000

# Bike capacity
QB = 80

# Satellite indicies, determined from satellite_selection.py
Satellites_ind = (7, 12)

def create_ave_demand_arr(Satellites_ind):
    """Creates 1D numpy array with average demand per customer"""
    
    demand = np.genfromtxt('demand.csv', delimiter=',')[1:17,1:]
    
    # Average demand per customer. Round up as we deal with whole units
    d_with_sats = np.mean(demand, axis=1)
    d_with_sats = np.ceil(d_with_sats)
    
    # Set satellite demands to 0
    demands_ave = []
    for i in range(len(d_with_sats)):
        if i not in Satellites_ind:
            demands_ave.append(d_with_sats[i])
        else:
            demands_ave.append(0)
    return demands_ave

D = create_ave_demand_arr(Satellites_ind)  # Demand at j

# Time from i to j by bike
P = np.genfromtxt('time_matrix_bike.csv', delimiter=',')  

# ---- Sets ---- 
num_nodes = 16
num_bikes = 5
Nodes = range(num_nodes)  # Set of non-satellites nodes, except bike depot
Bikes = range(num_bikes)

model = gp.Model("Design") # Make Gurobi model

# ---- Initialize variables

# Travel from node i to j for bike b
x = model.addVars(num_nodes, num_nodes, num_bikes, vtype=GRB.BINARY, name='x')            

# ---- Add constraints

# Bike exits all nodes that it enters
model.addConstrs(gp.quicksum(x[i,j,b] for j in Nodes if i != j)
                == gp.quicksum(x[j,i,b] for j in Nodes if i != j)
                for i in Nodes
                for b in Bikes)

# Every non-satellite node is entered exactly once
model.addConstrs(gp.quicksum(x[i,j,b] for i in Nodes for b in Bikes 
                            if (i != j))
                >= 1 for j in Nodes)        

# Every bike leaves depot  
model.addConstrs(gp.quicksum(x[0,j,b] for j in Nodes if j > 0) 
                >= 1 for b in Bikes)
    
    
# Bike capacity limit
model.addConstrs(gp.quicksum(D[node_map[j]]*x[i,j,b] 
                             for i in Nodes 
                             for j in Nodes 
                             if i != j)
                 <= QB for b in Bikes)    


# Subtour elimination constraints
u = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, lb=1, ub=num_nodes-1, 
                  name="u")
model.addConstrs(u[i] - u[j] + num_nodes * x[i, j, k] <= num_nodes - 2 
                  for i in range(1, num_nodes) 
                  for j in range(1, num_nodes) 
                  for k in range(num_bikes) if i != j)

# ---- Set objective

obj = gp.LinExpr()

# Obj function will minimize cost, but without taking into account time or 
# deliveries from the Depot yet, cost is minimal so we want to minimize
# time on bikes. So, we assign a cost to each minute ridden of €150/120

# Add cost for distance
for i in Nodes:
    i_map = node_map[i]
    for j in Nodes:
        j_map = node_map[j]
        if i != j:
            for b in Bikes:
                obj.addTerms(P[i_map, j_map], x[i, j, b])
    
model.setObjective(obj, GRB.MINIMIZE)

model.optimize()
#%%

import plots as pl

x_values = np.zeros((len(Nodes), len(Nodes), len(Bikes)))

# Fill in the values from the flattened array
for i in Nodes:
    for j in Nodes:
        for b in Bikes:
            if i != j:
                x_values[i, j, b] = x[i, j, b].X
                
pl.plot_routes(x_values)

np.save('x_values3.npy', x_values)
