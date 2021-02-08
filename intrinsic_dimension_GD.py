"""
Created on Sat Feb  6 15:25:21 2021

@author: Aslan

Adapted from: https://github.com/dgranata/Intrinsic-Dimension
"""

# Comments and questions can be send to daniele.granata@gmail.com

#  In case you find the code useful, please cite 
#  Daniele Granata, Vincenzo Carnevale  
#  "Accurate estimation of intrinsic dimension using graph distances: unraveling the geometric complexity of datasets" 
#  Scientific Report, 6, 31377 (2016)
#  https://www.nature.com/articles/srep31377

import numpy as np
from scipy.optimize import curve_fit
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.utils.graph import graph_shortest_path

def func(x, a, b, c):
    return a * np.log(np.sin(x / 1 * np.pi / 2.))
         
def func2(x, a):
    return -a / 2 * (x - 1) ** 2

def func3(x, a, b, c):
    return np.exp(c) * np.sin(x / b * np.pi / 2.) ** a

def estimate_GD(data, me = 'euclidean', n_neighbors = 100, radius = 0, n_bins = 50, rmax = 0, rmin = -10):
             
    ###1 Computing geodesic distance on connected points of the data
    
    if radius > 0.:
        A = radius_neighbors_graph(data, radius, metric = me, mode = 'distance')
        C = graph_shortest_path(A, directed = False)
    else:
        A = kneighbors_graph(data, n_neighbors, metric = me, mode = 'distance')
        C = graph_shortest_path(A, directed = False)
        radius = A.max()

    C = np.asmatrix(C)
    connect = np.zeros(C.shape[0])
    conn = np.zeros(C.shape[0])
     
    for i in range(0, C.shape[0]):
        conn_points = np.count_nonzero(C[i])
        conn[i] = conn_points
         
        if conn_points > C.shape[0] / 2.:
            connect[i] = 1
        else:
            C[i] = 0
     
    indices = np.nonzero(np.triu(C, 1))
    dist_list = np.asarray(C[indices])[-1]

    h = np.histogram(dist_list, n_bins)
    dx = h[1][1]- h[1][0]
    
    distr_x = []
    distr_y = []

    avg = np.mean(dist_list)
    std = np.std(dist_list)

    if rmax > 0:
        avg = rmax
        std = min(std, rmax) 
    else:
        mm = np.argmax(h[0])
        rmax = h[1][mm] + dx / 2

    if rmax == -1:
        avg = rmax
        std = min(std, rmax)
    
    tmp = 1000000
    
    if (rmin >= 0):
        tmp = rmin
    elif (rmin == -1):
        tmp = rmax - std
      
    ###2 Finding actual rmax and std to define fitting interval [rmin; rM] 
    
    distr_x = h[1][0:n_bins] + dx / 2
    distr_y = h[0][0:n_bins]
    
    res = np.empty(25)
    left_distr_x = np.empty(n_bins)
    left_distr_y = np.empty(n_bins)

    left_distr_x = distr_x[np.logical_and(np.logical_and(distr_x[:] > rmax - std, distr_x[:] < rmax + std / 2.), distr_y[:] > 0.000001)]
    left_distr_y = np.log(distr_y[np.logical_and(np.logical_and(distr_x[:] > rmax - std, distr_x[:] < rmax + std / 2.), distr_y[:] > 0.000001)])

    coeff = np.polyfit(left_distr_x, left_distr_y, 2, full = 'False')    
    a0 = coeff[0][0]
    b0 = coeff[0][1]
    c0 = coeff[0][2]
      
    rmax_old = rmax
    std_old = std
    rmax = -b0 / a0 / 2.
    
    if (rmax > 0):
        rmax = rmax   
    if a0 < 0 and np.fabs(rmax - rmax_old) < (std_old / 2 + dx):
        std = np.sqrt(-1 / a0 / 2.)
    else:
        rmax = avg
        std = std_old

    left_distr_x = distr_x[np.logical_and(distr_y[:] > 0.000001, np.logical_and(distr_x[:] > rmax - std, distr_x[:] < rmax + std / 2. + dx))]
    left_distr_y = np.log(distr_y[np.logical_and(distr_y[:] > 0.000001, np.logical_and(distr_x[:] > rmax - std, distr_x[:] < rmax + std / 2. + dx))])

    coeff = np.polyfit(left_distr_x, left_distr_y, 2, full = 'False')
    a = coeff[0][0]
    b = coeff[0][1]
    c = coeff[0][2]
    
    rmax_old = rmax
    std_old = std
    
    if a < 0.:
        rmax = -b / a / 2. 
        std = np.sqrt(-1 / a / 2.)
    
    rmin = max(rmax - 2 * std - dx / 2, 0.)
    
    if (rmin >= 0): 
        rmin = rmin
    elif (rmin < radius and rmin != -1): 
        rmin = radius
          
    rM = rmax + dx / 4

    if (np.fabs(rmax - rmax_old) > std_old / 4 + dx):
        rmax = rmax_old
        a = a0
        b = b0
        c = c0

        if (rmin >= 0):
            rmin = rmin
        elif (rmin < radius and rmin != -1):
            rmin = radius
        
        rM = rmax + dx / 4

    ###3 Gaussian fitting to determine ratio R
    
    left_distr_x = distr_x[np.logical_and(np.logical_and(distr_x[:] > rmin, distr_x[:] <= rM), distr_y[:] > 0.000001)] / rmax
    left_distr_y = np.log(distr_y[np.logical_and(np.logical_and(distr_x[:] > rmin, distr_x[:] <= rM ), distr_y[:] > 0.000001)]) - (4 * a * c - b ** 2) / 4. / a

    fit = curve_fit(func2, left_distr_x, left_distr_y)
    ratio = np.sqrt(fit[0][0])
    y1 = func2(left_distr_x, fit[0][0])

    ###4 Geodesics D-Hypersphere Distribution Fitting to determine Dfit

    fit = curve_fit(func, left_distr_x, left_distr_y)
    Dfit = (fit[0][0]) + 1

    y2 = func(left_distr_x, fit[0][0], fit[0][1], fit[0][2])
    
    ###5 Determination of Dmin
    
    for D in range(1, 26):
        y = (func(left_distr_x, D - 1, 1, 0))
        
        for i in range(0, len(y)):
            res[D - 1] = np.linalg.norm((y) - (left_distr_y)) / np.sqrt(len(y))

    Dmin = np.argmax(-res) + 1

    y = func(left_distr_x, Dmin - 1, fit[0][1], 0)
    
    return Dfit, Dmin