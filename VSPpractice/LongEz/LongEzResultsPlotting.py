# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30

Long Ez model CL, CD w/ stall model

@author: NASSAS
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# need to do to import from one folder above
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vspaero_processing_funcs import extract_vspaero_data
from scipy.interpolate import interp1d

#%% Taper = 1.0, AR = 3.0, b = 5.0 ft, c = 1.6666 ft
data_1 = extract_vspaero_data('LongEzAeroResults.csv')

fig, ax = plt.subplots(figsize = (6, 4), dpi = 800)
CD = data_1['CDtot'].values
CL = data_1['CLtot'].values
ax.plot(data_1['CLtot'], '-o', color = '#cc0000', markersize = 3, label = '$\\lambda = 1.0$, AR = 3, b = 5 ft')

plt.xlabel('$\\alpha$ ($^\\circ$)')
plt.ylabel('$C_L$')
plt.grid()
plt.show()


#%% CD plot
fig, ax = plt.subplots(figsize = (6, 4), dpi = 800)
CD = data_1['CDtot'].values
CL = data_1['CLtot'].values
ax.plot(data_1['CDtot'], '-o', color = '#cc0000', markersize = 3, label = '$\\lambda = 1.0$, AR = 3, b = 5 ft')

plt.xlabel('$\\alpha$ ($^\\circ$)')
plt.ylabel('$C_D$')
plt.grid()
plt.show()

#%% Drag polar
fig, ax = plt.subplots(figsize = (6, 4), dpi = 800)
CD = data_1['CDtot'].values
CL = data_1['CLtot'].values
ax.plot(CD, CL, '-o', color = '#cc0000', markersize = 3, label = '$\\lambda = 1.0$, AR = 3, b = 5 ft')

plt.xlabel('$C_D$')
plt.ylabel('$C_L$')
plt.grid()
plt.show()
print(f'Minimum CD = {CD.min():.6f} at CL = {CL[CD.argmin()]:.6f}')

W = 5390 #N approx
# very approx ~CD for cruise
# flying at 84 m/s cruise (~0.23 M at 15k ft)
rho = 1.19 # kg/m^3
Sw = 81.3*0.3048*0.3048 # ft^2 --> m^2
V = 84 # m/s
CLreq = (2*W)/(rho*(V**2)*Sw)
print(f'CL required = {CLreq:.6f}')

CDreq = interp1d(CL, CD, kind = 'cubic')(CLreq)
print(f'Associated CD = {CDreq:.6f}, note: no LG, interference effects, etc')
# 0.5*rho*V^2*Sw*CL = W
# CL = 2*W / (rho * V^2 * Sw)

#%% CD vs CL plot
fig, ax = plt.subplots(figsize = (6, 4), dpi = 800)
CD = data_1['CDtot'].values
CL = data_1['CLtot'].values
ax.plot(CD, CL, '-o', color = '#cc0000', markersize = 3, label = '$\\lambda = 1.0$, AR = 3, b = 5 ft')

plt.xlabel('$C_D$')
plt.ylabel('$C_L$')
plt.grid()
plt.show()
