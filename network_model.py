# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:37:07 2023

@author: rhanusa
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# import plots as pl
from plots import nmap

# ---- Global variables ----
n = 14  # Number of customers
n_bar = 22  # Total nodes
m = 1  # Number of bikes available
Q = 80  # Bike capacity
Q_hat = 40  # Max inventory on bike before refill permitted
T = 120  # Max time per route

# Satellite indicies, determined from satellite_selection.py
Satellites = (7, 12)

def create_ave_demand_arr(Satellites):
    """Creates 1D numpy array with average demand per customer"""
    
    demand = np.genfromtxt('demand.csv', delimiter=',')[1:17,1:]
    
    # Average demand per customer. Round up as we deal with whole units
    d_with_sats = np.mean(demand, axis=1)
    d_with_sats = np.ceil(d_with_sats)
    
    # Set satellite demands to 0
    demands_ave = []
    for i in range(len(d_with_sats)):
        if i not in Satellites:
            demands_ave.append(d_with_sats[i])
        else:
            demands_ave.append(0)
    return demands_ave

# Arrays below must be referenced using nmap
q = create_ave_demand_arr(Satellites)  # Demand at nmap[j]
p = [10 if i in Satellites else 5 for i in range(len(q))] # Load/refil time

q_bar = [-Q if i in Satellites else q[i] for i in range(len(q))]
Q_bar = [Q if i in Satellites else Q - q[i]for i in range(len(q))]


# Time from i to j by bike
tau = np.genfromtxt('time_matrix_bike.csv', delimiter=',')  

T_array = []

for i in range(len(tau)-1): # Subtract 1 since last row of tau is (real) depot
    T_list = []
    for j in range(len(tau)-1):
        if i in Satellites:
            temp_list = [p[i] + tau[i,k] + p[k] + tau[k,0] for k in range(len(p))]
            T_list.append(T-tau[0,j] - min(temp_list))
        else:
            T_list.append(T - tau[0,j] - (p[i] + tau[i,0]))
            
    T_array.append(T_list)
    
# TODO: Check, should this really be the lower bound for t?
t_lb = max([2*tau[0,i] + p[i] for i in range(len(p)) if i not in Satellites])  

tj_min = [min([tau[0,i] + p[i] + tau[i,j] for i in range(len(p))]) for j in range(len(p))]

tj_max = [max([T - p[j] - tau[j,i] - p[i] - tau[i,0] for i in range(len(p))]) for j in range(len(p))]

# Below here we use node indexes, no nmap is needed when referring to
# x, t, y

# ---- Sets ---- 

N_a = range(15,22)  # Nodes where bikes can reload (depot/satellite)
N = range(22)  # All nodes
I = range(1,15)  # All customer nodes, where a delivery must occur
I0 = range(15)  # Customer nodes plus depot
I_F = range(1,22)

model = gp.Model("Design") # Make Gurobi model

# ---- Initialize variables

# Travel from node i to j for bike b
# x = model.addVars(num_nodes, num_nodes, num_bikes, vtype=GRB.BINARY, name='x')            
x = model.addVars(n_bar, n_bar, vtype=GRB.BINARY, name='x') 

t = model.addVars(n_bar, vtype=GRB.CONTINUOUS, lb=t_lb, ub=T, name='t')

y = model.addVars(n_bar, vtype=GRB.CONTINUOUS, lb=0, ub=Q, name = 'y')

# ---- Add constraints

model.addConstrs(gp.quicksum(x[i,j] for j in N if i !=j) == 1 for i in I)

model.addConstrs(gp.quicksum(x[i,j] for j in I if i !=j) <= 1 for i in N_a)

# TODO: try restricting j to 1:21?
model.addConstrs(gp.quicksum(x[j,i] for i in N if i != j) 
                 - gp.quicksum(x[i,j] for i in N if i != j)
                 == 0 for j in N)

model.addConstr(gp.quicksum(x[0,j] for j in I) <= m)

model.addConstrs(t[j] >= t[i] + tau[nmap[i],nmap[j]]*x[i,j] 
                 - T_array[nmap[i]][nmap[j]] * (1-x[i,j]) 
                 for i in I_F for j in N if i != j)

model.addConstrs(t[j] >= tau[nmap[0], nmap[j]] for j in I)

model.addConstrs(t[j] <= T - (p[nmap[j]] + tau[nmap[j], nmap[0]]) for j in I)

# 9.1
model.addConstrs(t[j] >= tj_min[nmap[j]] for j in N_a)

model.addConstrs(t[j] <= tj_max[nmap[j]] for j in N_a)

#10
model.addConstrs(y[j] <= y[i] - q_bar[nmap[i]]*x[i,j] + Q_bar[nmap[i]]*(1-x[i,j])
                 for i in I_F for j in N if i != j)

model.addConstrs(y[j] >= q[nmap[j]] for j in I)

model.addConstrs(y[j] <= Q_hat for j in N_a)


# ---- Set objective

obj = gp.LinExpr()


for i in N:
    for j in N:
        if i != j:
            obj.addTerms(tau[nmap[i], nmap[j]], x[i, j])

    
model.setObjective(obj, GRB.MINIMIZE)

model.optimize()
#%%

import plots as pl

x_values = np.zeros((len(N), len(N)))

# Fill in the values from the flattened array
for i in N:
    for j in N:
        if i != j:
            x_values[i, j] = x[i, j].X
                
pl.plot_routes(x_values)

np.save('x_values4.npy', x_values)
