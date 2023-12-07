# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 07:38:44 2023

@author: rhanusa
"""
import numpy as np

from network_model import create_ave_demand_arr, Satellites
from plots import make_node_routes

satellite_nodes = (0, 15)

def create_std_demand_arr(Satellites):
    """Creates 1D numpy array with stdev demand per customer"""
    
    demand = np.genfromtxt('demand.csv', delimiter=',')[1:17,1:]
    
    # Stdev of demand per customer. Round up as we deal with whole units
    d_with_sats = np.std(demand, axis=1)
    # d_with_sats = np.ceil(d_with_sats)
    
    # Set satellite demands to 0
    demands_stdev = []
    for i in range(len(d_with_sats)):
        if i not in Satellites:
            demands_stdev.append(d_with_sats[i])
        else:
            demands_stdev.append(0)
            
    return demands_stdev

def create_rand_demand_arr(Satellites, seed):
    np.random.seed(seed)
    demands_ave = create_ave_demand_arr(Satellites)
    demands_stdev = create_std_demand_arr(Satellites)
    demands_stdev = [np.random.normal(ave, std) for ave, std in zip(demands_ave, demands_stdev)]
    return np.ceil(demands_stdev)

def calc_service_level():
    num_scenarios = 100
    
    unfulfilled_demand_tally = 0
    for seed in range(num_scenarios):
        demands_rand = create_rand_demand_arr(Satellites, seed)
    
        x_values = np.load('x_values4.npy')
        customer_routes = make_node_routes(x_values, satellite_nodes)
    
        # stocks = []
        for route in customer_routes:
            route_no_sat = [cust for cust in route if cust not in Satellites]
            # stock = [80]
            stock = 80 #
            for customer in route_no_sat:
                # remaining_stock = stock[-1]-demand_rand[customer]
                stock -= demands_rand[customer]
                # if remaining_stock < 0:
                if stock < 0: #
                    unfulfilled_demand_tally += 1
                # stock.append(remaining_stock)
            # stocks.append(stock)
        
    return 1-unfulfilled_demand_tally/(len(demands_rand)*num_scenarios)

service_level = calc_service_level()

