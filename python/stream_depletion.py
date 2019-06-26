# -*- coding: utf-8 -*-

from scipy.special import erfc
from math import sqrt
import numpy as np
#import matplotlib.pyplot as plt

#-Authorship information-###################################################################
__author__ = 'Wilco Terink'
__copyright__ = 'Wilco Terink'
__version__ = '1.0'
__email__ = 'wilco.terink@ecan.govt.nz'
__date__ ='June 2019'
############################################################################################

def Theis(T, S, L, q, d):
    '''
    Calculates the stream depletion rate in l/s given a constant pumping rate during x days. Single values as input.
    Returns:
        - sdf: Stream Depletion Factor
        - connection: %
        - SD: Stream Depletion Rate (L3/T) - Unit of SD depends on input unit of average pumping rate; e.g. input l/s = output in l/s, input in m3/d = output in m3/d 
    Input:
        - Transmissivity (m2/d)
        - Storage coefficient (-)
        - Separation distance (m)
        - Average pumping rate (L3/T)
        - Days of pumping with the average rate (d)
    '''

    sdf = L**2*S/T
    connection = erfc(sqrt(sdf/(4*d)))
    SD = connection * q 
    return sdf, connection, SD

#-SD for on/off pumping at various rates
def SD(L, S, T, Qpump):
    '''
    Calculates stream depletion in same unit as Qpump (numpy array)
    Input:
        - T = Transmissivity (m2/d)
        - S = Storage coefficient (-)
        - L = Separation distance (m)
        - Qpump = daily pumped volume
    Returns:
        - sd_matrix = stream depletion array (1-dimensional array) with same length as Qpump 
    '''
    
    sdf = (L**2)*S/T
    t = np.arange(1,len(Qpump)+1,1)

    ###-Calculate SD pumping going on and off and variable pumping rates
    dQ = np.zeros(len(t))
    sd_matrix = np.zeros([len(t), len(t)])
    for i in t:
        ix = np.argwhere(t==i)[0][0]
        if ix==0:
            dQ[ix] = Qpump[ix]
        else:
            dQ[ix] = Qpump[ix] - Qpump[ix-1]
        for j in t:
            if j>=i:
                jx = np.argwhere(t==j)[0][0]
                y = erfc(sqrt(sdf/(4*(j-i+1))))
                y = y * dQ[ix]
                sd_matrix[ix,jx] = y
    #-super position the individual curves
    sd_matrix = np.sum(sd_matrix, axis=0)
    

    return sd_matrix


    
# #-test values
# Distance = 440
# T_Estimate = 1100
# S = 0.100000001
# qpump = np.random.rand(50)*100
# 
# 
# sd = SD(Distance, S, T_Estimate, qpump)



# #-make a figure of the scenario with average pumping for the entire period, and on/off pumping at different rates
# plt.figure(facecolor='#FFFFFF')
# t = np.arange(1,len(sd)+1,1)
# lines = plt.plot(t,sd)
# plt.xlabel('Time')
# plt.ylabel('Stream depletion [l/s]')
# plt.grid(True)
# plt.legend(['On/Off'])
# plt.show(block=True)