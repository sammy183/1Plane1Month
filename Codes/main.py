# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 17:45:34 2025

Testing out my codes

@author: NASSAS
"""
from RaymerCh3 import *
import numpy as np

#%% Using raymer ch 3's crude methods

# checking that it works with the raymer example (pg 48)
Wcrew = 800 # lbf
Wpayload = 10000 # lbf
Swet_Sref = 5.5
AR = 7

# speed = 0.6 mach 
# I would like a function that gives V in fps or m/s for a provided altitude and M for now just use 
V = 596.9 # fps 

mission_profile = [['cruise', 1500, V], 
                   ['loiter', 3, V], 
                   ['cruise', 1500, V], 
                   ['loiter', 0.3333, V]]

design = AircraftV0(AR, Swet_Sref, Wcrew, Wpayload)
design.Type('Military Cargo-Bomber', 'military jets')
design.Propulsion('high-bypass turbofan')
W0 = design.W0calc(mission_profile)
print(W0) # estimated weight in lbs
design.RangeStudy(500, 2000, mission_profile)

# import scipy.optimize as sp
# import numpy as np
# # checking raymer example
# def W0func(W0):
#     W0calc = 10800/(1-0.3773-0.93*(W0**-0.07))
#     return(W0calc-W0)
# print(sp.root(W0func, 50000))
# lol raymer used different We/W0 values that made his box 3.1 example look better (perhaps a textbook edition problem)

#%% Testing for long range buisness jet
Wcrew = 800
Wpayload = 3000
Swet_Sref = 8
AR = 8 

V = 905
mission_profile = [['cruise', 5000, V],
                   ['loiter', 0.333, V]]
design = AircraftV0(AR, Swet_Sref, Wcrew, Wpayload)
design.Type('Business Jet', 'civil jets')
design.Propulsion('high-bypass turbofan')
W0 = design.W0calc(mission_profile)
print(W0) # estimated weight in lbs
design.RangeStudy(500, 8000, mission_profile)
# yeah this method is very insufficient, it says you can't get more than 5000 NM range with a buisness jet
# although the weight fractions are somewhat close throughout, 
# it becomes a problem when we/w0 doesn't decrease faster than wf/w0 grows (likely due to the crude modeling)


