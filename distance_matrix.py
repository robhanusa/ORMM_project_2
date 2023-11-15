# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:20:52 2023

@author: rhanusa
"""
import googlemaps
import numpy as np

LOCS = [
        'Meelstraat 2, 9000 Gent',
        'Dobbelslot 2, 9000 Gent',
        'Tolhuislaan 142, 9000 Gent',
        'Clarissenstraat 1, 9000 Gent',
        'Brabantdam 73, 9000 Gent',
        'Kasteellaan 74, 9000 Gent',
        'F. Rooseveltlaan 505, 9000 Gent',
        'Nederkouter 123, 9000 Gent',
        'Lammerstraat 37, 9000 Gent',
        'Sint-Michielsstraat 15, 9000 Gent',
        'Annonciadenstraat 21, 9000 Gent',
        'Ekkergemstraat 56, 9000 Gent',
        'Kettingstraat 94, 9000 Gent',
        'Bevrijdingslaan 154, 9000 Gent',
        'Wondelgemstraat87, 9000 Gent',
        'Noordstraat 38, 9000 Gent',
        'Oude-Abdijstraat 100, 9031 Drongen'
        ]

key = 'Your googlemaps API key here'
gmaps = googlemaps.Client(key=key)

#%%

def create_distance_matricies(locs, mode):

    # Break this down into 4 requests due to the limits on max elements per request
    dm1 = gmaps.distance_matrix(origins=locs,
                                destinations=locs[:5],
                                mode=mode,
                                units='metric',
                                region='Belgium')
    
    dm2 = gmaps.distance_matrix(origins=locs,
                                destinations=locs[5:10],
                                mode=mode,
                                units='metric',
                                region='Belgium')
    
    dm3 = gmaps.distance_matrix(origins=locs,
                                destinations=locs[10:15],
                                mode=mode,
                                units='metric',
                                region='Belgium')
    
    dm4 = gmaps.distance_matrix(origins=locs,
                                destinations=locs[15:],
                                mode=mode,
                                units='metric',
                                region='Belgium')

    return (dm1, dm2, dm3, dm4)


def convert_to_csv_matrix(locs, dms):
    dist_matrix = np.zeros([17,17])
    time_matrix = np.zeros([17,17])
    
    for origin in range(len(locs)):
        
        for count, dm in enumerate(dms):
            num_destinations = len(dm['rows'][origin]['elements'])
            
            for i in range(num_destinations):
                dist = dm['rows'][origin]['elements'][i]['distance']['value']/1000
                duration = dm['rows'][origin]['elements'][i]['duration']['value']/60
                
                destination = count*5 + i
                
                dist_matrix[origin][destination] = dist
                time_matrix[origin][destination] = duration * 2
                
    np.savetxt('distance_matrix_bike.csv',dist_matrix, delimiter=',')           
    np.savetxt('time_matrix_bike.csv',time_matrix, delimiter=',') 


#%%

mode = 'bicycling'
dms = create_distance_matricies(LOCS, mode)
convert_to_csv_matrix(LOCS, dms)
    