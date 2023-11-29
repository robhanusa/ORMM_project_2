# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 08:48:05 2023

@author: rhanusa
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Approximate number of minutes all satellites should be able to receive 
# average total demand from the depot.
load_cycle = 90

distance_matrix = np.genfromtxt('distance_matrix.csv', delimiter=',')

time_matrix = np.genfromtxt('time_matrix.csv', delimiter=',')

# Map satellite indicies to distance_matric indicies
satellite_map = {0: 1,
                 1: 3,
                 2: 7,
                 3: 12}

# Distance from depot for satellite candidates
sat_distance = [distance_matrix[-1][i] for i in satellite_map.values()]

# Time from depot to satellite candidates
sat_time = [time_matrix[-1][i] for i in satellite_map.values()]

# Round trip cost per van trip
trip_cost = [2*.8*sat_distance[i]+100+25 for i in range(len(sat_distance))]

# Daily satellite cost
sat_cost = [60, 120, 130, 50]

# Satellite capacity
cap = [120, 150, 180, 100]

# (Approximate) number of possible trips per load_cycle per satellite
max_trips = [round(load_cycle/(sat_time[i]*2)) for i in range(len(sat_time))]

num_satellites = 4
satellites = range(num_satellites)

model = gp.Model("Satellites") # Make Gurobi model

# ---- Initialize variables

# Whether or not a node is chosen as a satellite
x = model.addVars(num_satellites, vtype=GRB.BINARY, name='x')

# Number of trips to each satellite
y = model.addVars(num_satellites, vtype=GRB.INTEGER, name='y')

# ---- Add constraints

# Ensure that all trips to satellites can supply the average demand
model.addConstr(gp.quicksum(y[i]*cap[i] for i in satellites) >= 390)

# Max satellites is 3
model.addConstr(gp.quicksum(x[i] for i in satellites) <= 3)

# Trips from depot to satellite can only occur if satellite is chosen,
# And number of trips to each depot cannot exceed maximum
model.addConstrs(y[i] <= x[i]*max_trips[i] for i in satellites)


# ---- Set objective

obj = gp.LinExpr()

obj.addTerms(sat_cost, [x[i] for i in satellites])
obj.addTerms(trip_cost, [y[i] for i in satellites])

model.setObjective(obj, GRB.MINIMIZE)

#%% ---- Optimize model
model.optimize()

#%% Extract variable values

x_values = []
y_values = []

for i in satellites:
    x_values.append(x[i].X)
    y_values.append(y[i].X)
    
# With a 60 minute window for deliveries, objective is €642, using satellites
# 0, 2, and 3
# With a 120 minute window for deliveries, objective is €576, and only
# satellite 3 is used
# With a 90 minute window, objective is €582, and satellites 2 and 3 are used.
# Let's go with the 90 minute window, as this simulation is based on only 
# averages with no variability, so it would likely be the smart decision to 
# not only go with the minimum capacity satellite, as this decision only saves 
# €6, but will give significantly less flexibility.