# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:37:07 2023

@author: rhanusa
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# ---- Global variables ----
M = 1000

# Bike capacity (plus 20 as the lowest satellite demand, to account for the 
# fact that the bike doesn't have to carry satellite demand)
QB = 80 + 20 

def create_ave_demand_arr():
    """Creates 1D numpy array with average demand per customer"""
    
    demand = np.genfromtxt('demand.csv', delimiter=',')[1:17,1:]
    
    # Average demand per customer. Round up as we deal with whole units
    demands_ave = np.mean(demand, axis=1)
    
    return np.ceil(demands_ave)

D = create_ave_demand_arr()  # Demand at j

# Time from i to j by bike
P = np.genfromtxt('time_matrix_bike.csv', delimiter=',')  

# Cost of using a node as a satellite
C = (60, 120, 130, 50)


# ---- Sets ---- 
Nodes = range(16)  # Set of customers
Satellites = range(4)  # Set of possible satellites
Satellites_ind = (1, 3, 7, 12) # Set of possible satellites, adjusted for index
Bikes = range(6)

model = gp.Model("Design") # Make Gurobi model

# ---- Variables ----
x = {}  # Travel from node i to j for bike b
a = {}  # Whether node k is chosen as a satellie
w = {}  # Whether bike b leaves from satellite k


# ---- Initialize variables

# x[i,j,b]
for i in Nodes:
    for j in Nodes:
        for b in Bikes:
            if i != j:
                x[i,j,b] = model.addVar(vtype=GRB.BINARY, 
                                        name=f'x[{i},{j},{b}]')
                
# a[k]
for k in Satellites:
    a[k] = model.addVar(vtype=GRB.BINARY, name=f'a[{k}]')
    
    
# w[b,k]
for b in Bikes:
    for k in Satellites:
        w[b,k] = model.addVar(vtype=GRB.BINARY, name=f'w[{b},{k}]')
        

# ---- Add constraints

# Maximum of 3 satellites
model.addConstr(gp.quicksum(a[k] for k in Satellites) <=3)

# Bike cannot be assigned to a satellite if the satellite is not chosen
for k in Satellites:
    model.addConstr	(gp.quicksum(w[b,k] for b in Bikes) <= M*a[k])

# Bike exits all nodes that it enters
for i in Nodes:
    for b in Bikes:
        model.addConstr(gp.quicksum(x[i,j,b] for j in Nodes if i != j)
                        == gp.quicksum(x[j,i,b] for j in Nodes if i != j))
        
# Every node is entered at least once
for j in Nodes:
    model.addConstr(gp.quicksum(x[i,j,b] for i in Nodes for b in Bikes if i != j)
                    >= 1)
        
# Every bike leaves a satellite. If a satellite isn't chosen, bike may leave 
# more than 1 satellite
for b in Bikes:
    model.addConstr(gp.quicksum(x[k,j,b] for k in Satellites_ind for j in Nodes
                                if k != j) >= 1)
    
# Bike capacity limit
# TODO: need to exclude demand at the satellites
for b in Bikes:
    model.addConstr(gp.quicksum(D[j]*x[i,j,b] for i in Nodes for j in Nodes 
                                if i != j) <= QB)
    
# Total demand on bike route must be <= current inventory at satellite 
# For now we assume M inventory
for b in Bikes:
    model.addConstr(gp.quicksum(D[j]*x[i,j,b] for i in Nodes for j in Nodes 
                                if i != j) 
                    <= gp.quicksum(M*w[b,k] for k in Satellites))
    

# ---- Set objective

obj = gp.LinExpr()

# Obj function will minimize cost, but without taking into account time or 
# deliveries from the Depot yet, cost is minimal so we want to minimize
# time on bikes. So, we assign a cost to each minute ridden of €150/120

# Add cost for distance
for i in Nodes:
    for j in Nodes:
        if i != j:
            for b in Bikes:
                obj.addTerms(P[i,j], x[i,j,b])

for k in Satellites:
    obj.addTerms(C[k], a[k])
    
model.setObjective(obj, GRB.MINIMIZE)

model.optimize()
#%%

x_values = np.zeros((len(Nodes), len(Nodes), len(Bikes)))

# Fill in the values from the flattened array
for i in Nodes:
    for j in Nodes:
        for b in Bikes:
            if i != j:
                x_values[i, j, b] = x[i, j, b].X
                
