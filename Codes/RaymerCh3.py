# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 14:42:25 2025

Implementing Raymer Ch 3 basics for W0 estimation and range trades

US customary units abbr --> Imp
Metric -->  SI

This is extremely crude. It was mostly written to force me to engage with all the equations and the very basics.
Eventually I'll fully pivot to much better python alternatives (i.e. aviary/suave/aerosandbox/nikolai methods)

@author: Sammy Nassau
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciopt


#%% 
class AircraftV0:
    def __init__(self, AR, Swet_Sref, Wcrew, Wpayload):
        ''' 
        AR: Aspect ratio (choose from initial guess) 
        Swet_Sref: Wetted Area / Planform Wing Area (estimate from sketch)
        '''
        self.units = 'Imp' # use US imperial units by default
        self.typename = False
        self.propulsion = False
        self.proptype = False
        self.LDmax = False
        
        self.AR = AR
        self.Swet_Sref = Swet_Sref
        self.Wcrew = Wcrew
        self.Wpayload = Wpayload

        # for We/W0 fractions
        self.a_imp = {'GA-Metal-Single':1.250,
                 'GA-Metal-PistonTwin':1.289,
                 'GA-Composite':0.878,
                 'Homebuilt-Metal':1.08,
                 'Homebuilt-Composite':0.766,
                 'Turboprop Transport':0.975,
                 'Jet Transport':1.272,
                 'Business Jet':0.777,
                 'Military Cargo-Bomber':0.737,
                 'Jet Fighter':2.392,
                 'Jet Trainer':2.432,
                 'Sailplane-Unpowered':0.827, 
                 'Sailplane-Powered':0.876, 
                 'Ultralight':0.7, 
                 'Agricultural':1.205, 
                 'Aerobatic':0.783, 
                 'Flying Boat':1.092, 
                 'UAV-Jet':0.612, 
                 'UAV-Prop':0.804, 
                 }
        
        self.a_SI = {'GA-Metal-Single':1.164,
                'GA-Metal-PistonTwin':1.210,
                'GA-Composite':0.844,
                'Homebuilt-Metal':1.015,
                'Homebuilt-Composite':0.736,
                'Turboprop Transport':0.937,
                'Jet Transport':1.202,
                'Business Jet':0.761,
                'Military Cargo-Bomber':0.712,
                'Jet Fighter':2.158,
                'Jet Trainer':2.160,
                'Sailplane-Unpowered':0.795, 
                'Sailplane-Powered':0.842, 
                'Ultralight':0.673, 
                'Agricultural':1.122, 
                'Aerobatic':0.769, 
                'Flying Boat':1.050, 
                'UAV-Jet':0.589, 
                'UAV-Prop':0.767, 
                }
       
        self.C_weest = {'GA-Metal-Single':-0.090,
            'GA-Metal-PistonTwin':-0.080,
            'GA-Composite':-0.050,
            'Homebuilt-Metal':-0.078,
            'Homebuilt-Composite':-0.05,
            'Turboprop Transport':-0.05,
            'Jet Transport':-0.072,
            'Business Jet':-0.027,
            'Military Cargo-Bomber':-0.043,
            'Jet Fighter':-0.130,
            'Jet Trainer':-0.150,
            'Sailplane-Unpowered':-0.05, 
            'Sailplane-Powered':-0.05, 
            'Ultralight':-0.05, 
            'Agricultural':-0.09, 
            'Aerobatic':-0.023, 
            'Flying Boat':-0.05, 
            'UAV-Jet':-0.05, 
            'UAV-Prop':-0.06, 
            }
        
        # specific fuel consumption (1/hr or mg/(N*s))
        self.C_cruise_imp = {'turbojet': 0.9, 
                             'low-bypass turbofan': 0.8,
                             'high-bypass turbofan': 0.5}
        self.C_cruise_SI = {'turbojet': 25.5, 
                             'low-bypass turbofan': 22.7,
                             'high-bypass turbofan': 14.1}
        self.C_loiter_imp = {'turbojet': 0.8, 
                             'low-bypass turbofan': 0.7,
                             'high-bypass turbofan': 0.4}
        self.C_loiter_SI = {'turbojet': 22.7, 
                             'low-bypass turbofan': 19.8,
                             'high-bypass turbofan': 11.3}

        # propeller specific fuel consumption lb/hr/bhp or mg/(W*s)
        self.Cbhp_cruise_imp = {'piston-prop fixed-pitch': 0.4, 
                             'piston-prop variable-pitch': 0.4,
                             'turboprop': 0.5}
        self.Cbhp_cruise_SI = {'piston-prop fixed-pitch': 0.068, 
                             'piston-prop variable-pitch': 0.068,
                             'turboprop': 0.085}
        self.Cbhp_loiter_imp = {'piston-prop fixed-pitch': 0.5, 
                             'piston-prop variable-pitch': 0.5,
                             'turboprop': 0.6}
        self.Cbhp_loiter_SI = {'piston-prop fixed-pitch': 0.085, 
                             'piston-prop variable-pitch': 0.085,
                             'turboprop': 0.101}
        
        # for initial L/Dmax calculation
        self.K_LD = {'civil jets':15.5,
                 'military jets':14,
                 'retractable landing gear propeller':11,
                 'fixed landing gear propeller':9,
                 'high aspect ratio aircraft':13,
                 'sailplanes':15,
                 }

    def Type(self, typename, LDtype):
        '''
        General types for empty weight:
            (note: GA = general aviation)
            GA-Metal-Single
            GA-Metal-PistonTwin
            GA-Composite
            Homebuilt-Metal
            Homebuilt-Composite
            Turboprop Transport
            Jet Transport
            Business Jet
            Military Cargo-Bomber
            Jet Fighter
            Jet Trainer
            Sailplane-Unpowered
            Sailplane-Powered
            Ultralight
            Agricultural
            Aerobatic
            Flying Boat
            UAV-Jet
            UAV-Prop
        
        Additional types for L/Dmax calc:
            civil jets
            military jets
            retractable LG props
            fixed LG props
            high AR 
            sailplanes
        
        Variable sweep factor of Kvs = 1.04 ignored (until I attempt to use it in a design)
        '''
   
        if typename not in self.C_weest.keys():
            print('Please select one of:')
            for key in self.C_weest.keys():
                print(f'{key:14}')
            raise ValueError('Error: Aircraft type not recognized!')
        
        self.typename = typename
        
        if LDtype not in self.K_LD.keys():
            print('Please select one of:')
            for a in self.K_LD.keys():
                print(f'{a:14}')
            raise ValueError('Error: L/D type not recognized!')
            
        self.LDtype = LDtype
        
    def units(self, unitname):
        '''
        Input: 'SI' or 'Imp'
        '''
        if unitname not in ['SI', 'Imp']:
            print('ERROR: Wrong unit label input, please choose SI or Imp')
            return()
        
        self.units = unitname
    
    def We_W0(self, W0):
        if self.typename == False:
            raise RuntimeError('Aircraft type has not been defined with .Type(typename)!')
        
        if self.units == 'Imp':
            a = self.a_imp[self.typename]
            Const = self.C_weest[self.typename]
        else:
            a = self.a_SI[self.typename]
            Const = self.C_weest[self.typename]
        return(a*W0**Const)
    
    def Propulsion(self, propname):
        '''
        Options are: 
            turbojet, low-bypass turbofan, high-bypass turbofan, 
            piston-prop fixed-pitch, piston-prop variable-pitch, turboprop
        '''
                
        if propname in self.C_cruise_imp.keys():
            self.proptype = 'jet'
            self.propulsion = propname
        elif propname in self.Cbhp_cruise_imp.keys():
            self.proptype = 'prop'
            self.propulsion = propname
        else:
            raise ValueError('Propulsion not recognized, please choose one of\nturbojet, low-bypass turbofan, high-bypass turbofan\npiston-prop fixed-pitch, piston-prop variable-pitch, turboprop')
    
    def SpecificFuelConsumption(self, segment_name, V):
        '''V is velocity in m/s (SI) or fps (imp)
        
        1 bhp = 550 ft*lb/s'''
        if segment_name not in ['cruise', 'loiter']:
            raise ValueError('''Segment name not recognized, choose 'cruise' or 'loiter' ''')
            
        if self.proptype == 'prop':
            if self.propulsion == 'piston-prop fixed-pitch' and segment_name == 'loiter':
                eta_p = 0.7
            else:
                eta_p = 0.8
                
            if self.units == 'Imp':
                if segment_name == 'cruise':
                    C = self.Cbhp_cruise_imp[self.propulsion]*(V/(550*eta_p)) # imp, prop powered, cruise
                else: # loiter
                    C = self.Cbhp_loiter_imp[self.propulsion]*(V/(550*eta_p)) # imp, prop powered, loiter
            else:
                if segment_name == 'cruise':
                    self.C = self.Cbhp_cruise_SI[self.propulsion]*(V/eta_p) # SI, prop powered, cruise
                else: # loiter
                    C = self.Cbhp_loiter_SI[self.propulsion]*(V/eta_p) # SI, prop powered, loiter
        else:
            if self.units == 'Imp':
                if segment_name == 'cruise':
                    C = self.C_cruise_imp[self.propulsion] # imp, jet powered, cruise
                else: # loiter
                    C = self.C_loiter_imp[self.propulsion] # imp, jet powered, loiter
            else:
                if segment_name == 'cruise':
                    C= self.C_cruise_SI[self.propulsion] # SI, jet powered, cruise
                else: # loiter
                    C = self.C_loiter_SI[self.propulsion] # SI, jet powered, loiter
        return(C)
    
    def LiftToDrag(self, AR, Swet_Sref, segment_name):
        '''
        AR = b**2/Sref
        Swet_Sref = Swet/Sref = Swet (est from sketch)/planform wing area (typically)
        
        LD types from below list:
            
        Raymer pg 42-43 (K_LD est):
            15.5 for civil jets
            14 for military jets
            11 for retractable LG props
            9 for fixed LG props
            13 for high AR 
            15 for sailplanes
        '''
        
        # often (if you've taken more than a day on this project), you will have a better estimate for L/D
        # TODO: add in a way to provide L/D directly for cruise/loiter segments
        if self.proptype == False:
            raise RuntimeError('Propulsion type not defined yet!')
        if self.LDtype == False:
            raise RuntimeError('L/D max type not defined yet!')
        
        self.LDmax = self.K_LD[self.LDtype]*np.sqrt(AR/Swet_Sref)
        
        if self.proptype == 'prop':
            if segment_name == 'cruise':
                return(self.LDmax)
            elif segment_name == 'loiter':
                return(0.866*self.LDmax)
        else:
            if segment_name == 'cruise':
                return(0.866*self.LDmax)
            elif segment_name == 'loiter':
                return(self.LDmax)        
            
    def Wi_Cruise(self, R, V, AR, Swet_Sref):
        '''
        Calculates W_i/W_i-1 for cruise
        
        units == imp --> R is range in nautical miles, V is velocity in fps
        units == SI --> R is range in kilometers, V is velocity in m/s
        '''
        
        if self.units == 'Imp':
            R_ft = 6076.12*R # convert NM to ft
            C = self.SpecificFuelConsumption('cruise', V)*(1/3600) # convert to 1/s
            LD = self.LiftToDrag(AR, Swet_Sref, 'cruise')
            Wi = np.exp(-(R_ft*C)/(V*LD))
            return(Wi)
        
        else: # metric
            R_m = R/1000 # convert to m
            C = self.SpecificFuelConsumption('cruise', V)/10e6 # convert to kg/(N*s) = kg/((kg*m/s^2)*s) = 1/(m/s) = s/m
            LD = self.LiftToDrag(AR, Swet_Sref, 'cruise')
            Wi = np.exp(-(R_m*C)/(V*LD))
        return(Wi)
    
    def Wi_Loiter(self, E, V, AR, Swet_Sref):
        '''
        Calculates W_i/W_i-1 for loiter
        
        E is endurance in hrs
        
        note: V doesn't have to be accurate
        '''
        E_s = E*3600
        
        if self.units == 'Imp':
            C = self.SpecificFuelConsumption('loiter', V)*(1/3600) # convert to lb/(lbf*s)
            LD = self.LiftToDrag(AR, Swet_Sref, 'loiter')
            Wi = np.exp(-(E_s*C)/LD)
            return(Wi)
        else: # metric
            C = self.SpecificFuelConsumption('loiter', V)/10e6 # convert to kg/(N*s) = kg/((kg*m/s^2)*s) = 1/(m/s) = s/m
            LD = self.LiftToDrag(AR, Swet_Sref, 'loiter')
            Wi = np.exp(-(E_s*C)/LD)
            return(Wi)
        
    def Wf_W0(self, mission_profile):
        '''
        mission profile is a list of lists of the form:
        [
            ['cruise', RANGE, V],
            ['loiter', hrs, V],
            ['cruise', RANGE, V],
            ['loiter', hrs, V]
        ]         
        
        if units are in SI, Range in kilometers, V in fps
        if in Imp, Range in nautical miles, V in m/s
        
        takeoff is assumed to be the first segment, climb is assumed to be second, and landing is assumed to be the last 
        '''
        Wi = [0.97] # warmup and takeoff
        Wi.append(0.985) # climb
        
        # iteration over mission profile to estimate Wi/Wi-1 for cruise/loiter segments
        if self.propulsion == False:
            raise RuntimeError('Please define propulsion with .Propulsion() first')
        
        for seg, val, Vspec in mission_profile:
            if seg == 'cruise':
                Wi.append(self.Wi_Cruise(val, Vspec, self.AR, self.Swet_Sref))
            elif seg == 'loiter':
                Wi.append(self.Wi_Loiter(val, Vspec, self.AR, self.Swet_Sref))
        
        Wi.append(0.995) # landing
        
        Wi = np.array(Wi)
        Wi = np.prod(Wi) # multiply the segment fractions together
    
        return(1.06*(1 - Wi))

    def W0calc(self, mission_profile):
        wfw0 = self.Wf_W0(mission_profile)
        # We/W0 is a function of W0

        # solve W0 = (Wcrew + Wpayload)/(1 - wfw0 - wew0)
        def W0func(W0, *args):
            self, wfw0 = args
            wew0 = self.We_W0(W0)
            if 1 < wfw0 + wew0:
                print(wfw0)
                # print(wew0)
                return(-1)
            W0calc = (self.Wcrew + self.Wpayload)/(1 - wfw0 - wew0)
            return(W0calc-W0)

        W0 = sciopt.root(W0func, 10000, args=(self, wfw0))
        return(W0.x[0])
    
    def RangeStudy(self, lowRange, highRange, mission_profile):
        '''
        R in NM or km depending on units
        '''
        ranges = np.linspace(lowRange, highRange, 15)
        weights = np.zeros(ranges.size)
        mission_profile_new = mission_profile.copy()
        for j, Rspec in enumerate(ranges):
            # construct new mission profile (assumes all cruise has same distance)
            for i, (seg, val, V) in enumerate(mission_profile):
                if seg == 'cruise':
                    mission_profile_new[i] = [seg, Rspec, V]
            # calculate weights
            weights[j] = self.W0calc(mission_profile_new)
        
        plt.figure(dpi = 1000)
        plt.plot(ranges, weights)
        plt.xlabel('Range (NM)')
        plt.ylabel('Weight (lbs)')
        plt.grid()
        plt.show()
        
        
        
        
        