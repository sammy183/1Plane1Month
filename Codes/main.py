# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 17:45:34 2025

Testing out my codes

@author: NASSAS
"""
from RaymerCh3 import *
import numpy as np
from ClassicalSizingFunctions import TW_WS
ftm = 0.3048
lbf_ft2_to_kg_m2 =  4.88243
ktom = 1/1.94384 # knot to m/s
ktofps = 1.68781

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
# W0 = design.W0calc(mission_profile)
# print(W0) # estimated weight in lbs
# design.RangeStudy(500, 2000, mission_profile)

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
# W0 = design.W0calc(mission_profile)
# # print(W0) # estimated weight in lbs
# design.RangeStudy(500, 8000, mission_profile)
# yeah this method is very insufficient, it says you can't get more than 5000 NM range with a buisness jet
# although the weight fractions are somewhat close throughout, 
# it becomes a problem when we/w0 doesn't decrease faster than wf/w0 grows (likely due to the crude modeling)

#%% T/W W/S plotting (OLD)
# AR = 3.0
# e = 0.85
# CDmin = 0.03
# eta_p = 0.6
# Design = TW_WS(AR, e, CDmin, eta_p)

# # all in m/s
# Vclimb = 30
# Vv = 7.62 # 1500 fpm
# n = 6
# Vturn = 25
# h = 0 # ALL AT SSL

# WS = np.linspace(15, 150) #kg/m2
# Design.WSrange(WS)
# Design.TW_climb(Vv, Vclimb, h)
# Design.TW_susturn(Vturn, h, n = n)

# dgr = 10*ftm # 200 ft
# takeoff_surface = 'dry concrete'
# CLto = 0.8
# CDto = 0.05 
# CLmax = 1.5
# Design.TW_takeoff(dgr, takeoff_surface, CLto, CDto, CLmax)
# Design.plot()


#%% following along with https://www.youtube.com/watch?v=qnspsMprpa8
# small aircraft for LSA category; 2 people carrying

## SPEEDS TO REMEMBER
# VEAS = VTAS*np.sqrt(rho/rho0)
# VCAS = VIAS + error = VEAS

#### Requirements: 
    # Vs <= 45 KCAS
    # Vt = 140 KTAS @ 10,000 ft --> Vc ~= 120 KCAS
    # Vv = 1500 fpm @ SSL
    # hmax >= 20,000 ft
    # dgr <= 800 ft @ SSL and standard atmosphere
    # bank angle >= 45 deg for sustained turn at cruise condition
    # MGTOW <= 1320 lbs

# NOTE: all T/W eqns use DIFFERENT qs
#   sustain turn: q from rho at 10,000 ft, V = 140 KTAS
#   takoeff: q from rho = 0 ft, V = Vlof/np.sqrt(2)


# given by problem
CDmin = 0.032
k = 0.04207

AR_e = 1/(np.pi*k)
e = 0.85 # assume to get the right k
AR = AR_e/0.85
eta_p = 0.7
Design = TW_WS(AR, e, CDmin)

# range of wing loadings to use
WS = np.linspace(1, 30) # lbf/ft2
Design.WSrange(WS)

# # Vs <= 45 KCAS
# Vs = 45*ktofps #KCAS (do I convert to EAS?)
# CLmax = 1.2 # no high lift
# Design.WSstall(Vs, CLmax)

# Vt = 140 KTAS @ 10,000 ft --> Vc ~= 120 KCAS
Vcruise = 140*ktofps #KTAS
Vturn = Vcruise
hturn = 10000 # ft to m
Design.TW_susturn(Vturn, hturn, phi = 45)

# Vv = 1500 fpm @ SSL (don't know what speed?)
Vv = 1500/60 #fpm to fps
Vclimb = 65*ktofps
h = 0 
Design.TW_climb(Vv, Vclimb, h)

# hmax >= 20000 ft
h = 20000
Vv = 100/60 # fpm to fps at service ceiling req for prop aircraft
Design.TW_ceiling(Vv, h)

# dgr <= 800 ft @ SSL
dgr = 800 # ft
takeoff_surface = 'dry concrete'
CLto = 0.5
CDto = 0.035
Vstall = 45*ktofps
Design.TW_takeoff(dgr, takeoff_surface, CLto, CDto, Vstall) ### THIS ONE IS OFF!!!

# cruise at 140 KTAS
hcruise = hturn
Design.TW_cruise(Vcruise, hcruise)

Design.plot()

