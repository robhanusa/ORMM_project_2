# -*- coding: utf-8 -*-
"""
Model adapted from:
Bard, Jonathan & Huang, Liu & Dror, Moshe & Jaillet, Patrick. (1998). 
A branch and cut algorithm for the VRP with satellite facilities. 
IIE Transactions. 30. 821-834. 10.1023/A:1007500200749. 
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from plots import nmap

# ---- Global variables ----
n = 14  # Number of customers
n_bar = 22  # Total nodes
m = 2  # Number of bikes available
Q = 60  # Bike capacity
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
p = [10 if i in Satellites else 5 for i in range(len(q))] # Load/refill time

q_bar = [-Q if i in Satellites else q[i] for i in range(len(q))]
Q_bar = [Q if i in Satellites else Q - q[i]for i in range(len(q))]


# Time from i to j by bike
tau = np.genfromtxt('time_matrix_bike.csv', delimiter=',')  

T_array = []

for i in range(len(tau)-1): # Subtract 1 since last row of tau is (real) depot
    T_list = []
    for j in range(len(tau)-1):
        if i in Satellites:
            temp_list = [p[i] + tau[i,k] + p[k] \
                         + min(tau[k, Satellites[0]], tau[k, Satellites[1]]) 
                         for k in range(len(p))]
            T_list.append(T-min(tau[Satellites[0],j], tau[Satellites[1],j]) \
                          - min(temp_list))
        else:
            T_list.append(T - min(tau[Satellites[0],j], tau[Satellites[1],j]) \
                          - (p[i] + min(tau[i, Satellites[0]], 
                                        tau[i, Satellites[1]])))
            
    T_array.append(T_list)
    
# Lower bound for t[0]
t_lb = max([2*min(tau[Satellites[0],i], tau[Satellites[1],i]) + p[i] 
            for i in range(len(p)) if i not in Satellites])  

tj_min = [min([min(tau[Satellites[0],i], tau[Satellites[1],i]) + p[i] + tau[i,j] 
               for i in range(len(p))]) for j in range(len(p))]

tj_max = [max([T - p[j] - tau[j,i] - p[i] \
               - min(tau[i, Satellites[0]], tau[i, Satellites[1]]) 
               for i in range(len(p))]) for j in range(1,len(p))]

# Below here we use node indexes, no nmap is needed when referring to
# x, t, y

# ---- Sets ---- 

S = (0,15)  # Nodes corresponding to a satellite
N_a = range(16,22)  # Nodes where bikes can reload (copies of satellites)
N = range(22)  # All nodes
I = range(1,15)  # All customer nodes, where a delivery must occur
I0 = range(16)  # Customer nodes plus satellites
I_F = [i for i in range(1,22) if i !=15]  # Nodes not including satellites

model = gp.Model("Design") # Make Gurobi model


# ---- Initialize variables

# Travel from node i to j for bike b           
x = model.addVars(n_bar, n_bar, vtype=GRB.BINARY, name='x') 

t = model.addVars(n_bar, vtype=GRB.CONTINUOUS, lb=0, ub=T, name='t')

y = model.addVars(n_bar, vtype=GRB.CONTINUOUS, lb=0, ub=Q, name = 'y')


# ---- Add constraints

# Ensure each customer has exactly 1 successor
model.addConstrs(gp.quicksum(x[i,j] for j in N if i !=j) == 1 for i in I)

# Each copy of a satellite has at most one successor
model.addConstrs(gp.quicksum(x[i,j] for j in I if i !=j) <= 1 for i in N_a)

# Each satellite has at most one successor
model.addConstrs(gp.quicksum(x[i,j] for j in I if i !=j) <= 1 for i in S)

# Number of arrivals at a node equal number of departures
model.addConstrs(gp.quicksum(x[j,i] for i in N if i != j) 
                 - gp.quicksum(x[i,j] for i in N if i != j)
                 == 0 for j in N)

# No more than m vehicles used (i.e. only m vehicles can start journey at a depot)
model.addConstr(gp.quicksum(x[i,j] for i in S for j in I) <= m)
# model.addConstr(gp.quicksum(x[j,0]+x[j,15] for j in I) <= m)

# Tracks the time a service begins at node j. Also functions to eliminate subtours
model.addConstrs(t[j] >= t[i] + tau[nmap[i],nmap[j]]*x[i,j] + p[nmap[j]] 
                 - T_array[nmap[i]][nmap[j]] * (1-x[i,j]) 
                 for i in I_F for j in N if i != j)

# Lower bound for arrival back at depot. Must be at least as long as visiting
# the farthest customer, dropping off the load, and returning to depot
model.addConstrs(t[s] >= t_lb for s in S)

# Lower bound for t at customer nodes. There must be enough time to get to the
# customer from the depot.
model.addConstrs(t[j] >= min(tau[Satellites[0], nmap[j]], 
                             tau[Satellites[1], nmap[j]]) 
                 for j in I)

# Upper bound for t at customer nodes. There must be enough time left for the
# bike to get back to the depot
model.addConstrs(t[j] <= T - (p[nmap[j]] \
                              + min(tau[nmap[j], Satellites[0]], 
                                    tau[nmap[j], Satellites[1]])) 
                 for j in I)

# Minimum amount of time to go from depot to customer, provide service, and go
# to a satellite.
model.addConstrs(t[j] >= tj_min[nmap[j]] for j in N_a)

# Latest time to depart satellite/depot j, visit a customer, and return to
# The depot arriving no later than T
model.addConstrs(t[j] <= tj_max[nmap[j]] for j in N_a)

# Tracks the load on a bike just prior to visiting node j
model.addConstrs(y[j] <= y[i] - q_bar[nmap[i]]*x[i,j] \
                 + Q_bar[nmap[i]]*(1-x[i,j])
                 for i in I_F for j in N if i != j)

# Load must be high enough to service the customer
model.addConstrs(y[j] >= q[nmap[j]] for j in I)

# Load must be under Q_hat if bike will stop at a depot/satellite
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
y_values = np.zeros(len(N))
t_values = np.zeros(len(N))

# Fill in the values from the flattened array
for i in N:
    y_values[i] = y[i].X
    t_values[i] = t[i].X
    for j in N:
        if i != j:
            x_values[i, j] = x[i, j].X
            
                
pl.plot_routes(x_values, S)

np.save('x_values4.npy', x_values)
np.save('t_values.npy', t_values)
