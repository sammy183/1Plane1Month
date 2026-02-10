# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 13:29:15 2025

Taper Results

@author: NASSAS
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# need to do to import from one folder above
import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from vspaero_processing_funcs import extract_vspaero_data

#%% Taper = 1.0, AR = 3.0, b = 5.0 ft, c = 1.6666 ft
data_1 = extract_vspaero_data('HersheyBar.csv')

fig, ax = plt.subplots(figsize = (6, 4), dpi = 800)
CD1 = data_1['CDtot'].values
CL1 = data_1['CLtot'].values
# ax.plot(data_1['L_D'], 'o', color = '#cc0000', markersize = 3)
ax.plot(CD1, CL1, '-o', color = '#cc0000', markersize = 3, label = '$\\lambda = 1.0$, AR = 3, b = 5 ft')


#%% Taper = 0.45, AR = 3.0, b = 5.0 ft
data_045 = extract_vspaero_data('Taper045_Sweep0.csv')
CD045 = data_045['CDtot'].values
CL045 = data_045['CLtot'].values
ax.plot(CD045, CL045, '-o', color = 'k', markersize = 3, label = '$\\lambda = 0.45$, AR = 3, b = 5 ft')

# ax.plot(data_045['CDtot'], 'o', color = 'k', markersize = 3)

#%% Taper = 0.45, Sweep = 10 deg, AR = 3.0, b = 5ft
data_045_s10 = extract_vspaero_data('Taper045_Sweep10.csv')
CD045_s10 = data_045_s10['CDtot'].values
CL045_s10 = data_045_s10['CLtot'].values
ax.plot(CD045_s10, CL045_s10, '-o', color = 'blue', markersize = 3, label = '$\\lambda = 0.45$, $\\Lambda$ = 10 deg, AR = 3, b = 5 ft')

# plt.ylabel('Induced Drag Coefficient ($C_{Di}$)')
plt.legend()
plt.ylabel('$C_L$')
plt.xlabel('$C_D$')
plt.grid()
plt.show()

#%% plotting
# ax.plot(data_1['L_D'], '-o', color = '#cc0000', markersize = 3, label = '$\\lambda = 1.0$, AR = 3, b = 5 ft')
# ax.plot(data_045['L_D'], '-o', color = 'k', markersize = 3, label = '$\\lambda = 0.45$, AR = 3, b = 5 ft')
# ax.plot(data_045_s10['L_D'], '-o', color = 'blue', markersize = 3, label = '$\\lambda = 0.45$, $\\Lambda$ = 10 deg, AR = 3, b = 5 ft')

# plt.xlabel('$\\alpha$ ($^\\circ$)')
# plt.ylabel('L/D')

fig, ax = plt.subplots(dpi = 800)
ax.plot(data_1['CLtot'], '-o', color = '#cc0000', markersize = 3, label = '$\\lambda = 1.0$, AR = 3, b = 5 ft')
ax.plot(data_045['CLtot'], '-o', color = 'k', markersize = 3, label = '$\\lambda = 0.45$, AR = 3, b = 5 ft')
ax.plot(data_045_s10['CLtot'], '-o', color = 'blue', markersize = 3, label = '$\\lambda = 0.45$, $\\Lambda$ = 10 deg, AR = 3, b = 5 ft')

plt.xlabel('$\\alpha$ ($^\\circ$)')
plt.ylabel('$C_L$')
plt.grid()
plt.show()

