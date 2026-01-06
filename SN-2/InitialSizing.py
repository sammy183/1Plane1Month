# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 16:30:47 2026

For the high altitude aircraft experimentation

Reqs: 
    100k ft for cruise and loiter
    Endurance: 24-48 hrs
    Range: 7000 Nm
    
Goals:
    Independantly understand why high-alitude aircraft favor ridiculously large ARs
    Refine completely unrealistic requirements
    Practice weight estimation and initial and optimized layouts without relevant historical data
        (then check with historical benchmarks to verify accuracy)
        
        
Nice find but only deals with up to 50k ft: 
    EFFECTS OF ALTITUDE ON TURBOJET ENGINE PERFORMANCE By William A. Fleming 
    https://ntrs.nasa.gov/api/citations/19930087116/downloads/19930087116.pdf
    1951
    
Booyah:
    TURBOJET PERFORMANCE AND OPERATION AT HIGH ALTITUDES WITH HYDROGEN AND JP -4 FUELS
    By W. A. Fleming, H. R. Kaufman, J. L. Harp, Jr., and L. J. Chelko 

    This specifically says JP-4 is only good to ~70-75 kft altitude
    
NASA ERAST Program developed reciprocating engines capable of ~85 kft with three-stage superchargers

@author: NASSAS
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects

from Codes import ClassicalSizingFunctions
from Codes import Atmosphere as atmosphere
ftm = 0.3048
lbfN = 4.44822
lbfft2_Nm2 = 47.880258888889

#%% Number 1 question is how does a turbojet perform at 100k ft?
# or more generally, with respect to altitude



#%% Initial check of Vreq based on stall + mach speeds

# plot: yaxis Altitude, xaxis Velocity, stall speed, mach 1

V = np.linspace(20, 350, 500) # m/s
h = np.linspace(0, 35, 500) # geometric altitude in km

atm = atmosphere.stdatm1976()
rho = atm.rho(h*1000) # density in kg/m3

# plt.plot(h, rho)
# plt.show()

plt.figure(figsize = (6, 4), dpi = 800)
CLmax = 0.9 # high CL airfoil 

target = 100000 # ft
target *= ftm
target /= 1000
plt.axhline(target, linestyle = '--', color = 'red', label = 'Target Altitude')


# GOAL: for a given wing loading, show the stall limit line (with CL assumption)
# IDEAL: want a contour plot of WS vs velocity and altitude
# CONFUSION: why is the contour not identical to a fixed, known W/S? 
# ANSWER: I'm a moron and did my algebra wrong with CLmax lol
X, Y = np.meshgrid(V, h)
newV, newRho = np.meshgrid(V, rho)  #np.logspace(start=0, stop=1.3, num = 4)
WS = (0.5*newRho*(V**2)*CLmax) / lbfft2_Nm2 # stall wing loadings
lines = plt.contour(X, Y, WS, levels = [2.5, 5.5, 9, 15], colors = 'blue')
plt.clabel(lines, fmt='%.1f psf')
lines.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
plt.plot([], [], 'b-', label='Stall Limits for W/S')
plt.plot([150, 200, 250], [target]*3, 's', color = '#cc0000', markersize = 5, label = 'Possible Design Points', zorder = 3)

# speed of sound change with altitude
T = atm.T(h*1000) # temperature change
newV, newT = np.meshgrid(V, T)
gamma = 1.4 # for air
R = 286 # m^2/(s^2*K) # for air
a = np.sqrt(gamma*R*newT)
M = X/a 
lines = plt.contour(X, Y, M, levels = [0.25, 0.5, 0.75], colors = 'k')
plt.plot([], [], 'k-', label='Mach Number')
plt.clabel(lines, fmt='M %.2f')
limitline = plt.contour(X, Y, M, levels = [1], colors = 'k')
limitline.set(path_effects = [patheffects.withTickedStroke(spacing = 10, angle = 135, length = 0.5)])
plt.clabel(limitline, fmt='M %.2f')

plt.xlabel('Velocity (m/s)')
plt.ylabel('Altitude (km)')
plt.legend(fontsize = 5)
plt.xlim([V.min(), V.max()])
# plt.title(f'For Sw = {Sw:.2f} m2, CLmax = {CLmax}, W = {W/lbfN:.0f} lbs')
plt.show()

# so our goal should be to increase Sw, increase CL, and decrease W
# how much of each is needed to reach that magical 100,000 ft? 
# Even discounting T = D propulsion issues and the buffet zone well before mach 1.0

#%% need Vstall <= 300 m/s at 100,000 ft
Vstallreq = 250

density = atm.rho(target*1000)
# print(density)
rho = 0.01710029158076083 # kg/m3

# take CLmax = 0.9
CLmax = 0.9
for Vs in [150, 200, 250]:
    WS = (rho*(Vs**2)*0.5)*CLmax
    # print(WS)
    print(f'W/S = {WS/lbfft2_Nm2:.2f} lbf/ft² for {Vs} m/s and 100 kft')

# W/S = 10.04 lbf/ft² (very low, U-2 was reportedly 40 lb/ft2)
# W/S = 3.62 lbf/ft² for 150 m/s and 100 kft
# W/S = 6.43 lbf/ft² for 200 m/s and 100 kft
# W/S = 10.04 lbf/ft² for 250 m/s and 100 kft


#%% Now we can proceed to the constraint analysis graphs



#%% this will also provide the list of codes I want


