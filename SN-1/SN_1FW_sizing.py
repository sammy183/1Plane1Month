# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 14:59:18 2025

RC aircraft with COTS propulsion components, <100Wh, <55lb MGTOW, 
    MAXIMIZED RANGE

@author: NASSAS
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects

lbfft2_kgm2 = 4.88243 # conversion factor from lbf/ft^2 to kg/m^3 #REVIEW UNITS LATER
lbkg = 1/2.205 # cardinal sin, ik ik
ftm = 0.3048

#fun sidetrack
# from datetime import datetime
# current_datetime = datetime.now()
# formatted_time = current_datetime.strftime("%H_%M_%S")

#%% new function for automated sensitivity analysis plotting
def automated_sensitivity(names, variables, base_array, normalization_array, func, metricname = False, save = False, norm = True):
    '''
    names must also correspond; it's important for plotting
    variables and normalization_array must correspond where variables are 1xn arrays of possibilities
    and normalization_array elements are scalars denoting best possible values'''
    fig, ax = plt.subplots(figsize = (6, 4), dpi = 1000)
    
    midvalue = func(base_array)
    for i, var in enumerate(variables):
        # need to feed the proper array/scalar based on i
        # i.e. for i = 1, I want L_D as 1xn, and then eta_b2s, eta_p, m as the normalization scalars
        # then for i = 2, L_D (scalar), eta_b2s (array), eta_p (scalar), m (scalar)
        varuses = list.copy(base_array)
        varuses[i] = variables[i]
        if norm != True:
            midvalue = 100
        ax.plot(var/base_array[i]*100, func(varuses)/midvalue*100, '--', label = names[i])
    
    plt.grid()
    plt.xlabel('% Change in Variables')
    if metricname and norm:
        plt.ylabel(f'% Change in {metricname}')
    elif metricname:
        plt.ylabel(f'{metricname}')
    else:
        plt.ylabel('% Change in Metric')
    plt.title('Sensitivity Analysis')
    plt.legend()
    if save and metricname:
        plt.savefig(f'{metricname}_sensitivity_analysis', dpi = 1000)
    
    plt.show()

#%% Start: sensitivity study

# primary range equation: R = 3.6*(L/D)*((Esb*eta_b2s*eta_p)/g)*(m_b/m) 
# Esb*m_b = 100 Wh (requirement)
num = 100

L_D_base = 20
eta_b2s_base = 0.93
eta_p_base = 0.8   
m_base = 3    # kg
base_array = [L_D_base, eta_b2s_base, eta_p_base, m_base]

L_D_max = 25
eta_b2s_max = 0.95
eta_p_max = 0.85
m_max = 2.5
norm_array = [L_D_max, eta_b2s_max, eta_p_max, m_max]

L_D = np.linspace(15, L_D_max, num)
eta_b2s = np.linspace(0.85, eta_b2s_max, num)
eta_p = np.linspace(0.7, eta_p_max, num)
m = np.linspace(2.5, 4, num)
var = [L_D, eta_b2s, eta_p, m]

names = ['L/D', r'$\eta_{b2s}$', r'$\eta_{p}$', 'm (kg)']

def RangeFunc(variables):
    '''
    each var can be a 1xn numpy array or a scalar
    this produces the required 1xn R arrays for plotting range as a sensitivity analysis'''
    L_D, eta_b2s, eta_p, m = variables
    g = 9.81    # kg/m^3
    E = 100     # Wh
    
    R = 3.6*L_D*(E*eta_b2s*eta_p)/(g*m)
    return(R)

# automated_sensitivity(names, var, base_array, norm_array, RangeFunc, metricname = 'Range (km)', norm = False, save=True)
# conclusions are pretty cut and dry: 
    # minimizing m and maximizing L/D are by far the most important and variable quantities

#%% T/W, W/S for dash 1 sizing
class TW_WS:
    def __init__(self, AR, e, CD0):
        '''AR = aspect ratio, e = oswalds efficiency, CD0 is zero-lift drag coefficient'''
        self.rho = 1.23 # kg/m3
        self.g = 9.807 # m/s2
        self.eta_p = 0.8
        self.AR = AR
        self.e = e
        self.CD0 = CD0
        self.k = 1/(np.pi*self.AR*self.e)
        self.TWs = [False, False, False, False, False] # sustained turn, cruise, climb, takeoff, landing
        self.stallreq = False
        self.rangereq = False

        # fake initializations so the plotting f-strings don't mess stuff up
        self.Vturn = -50.0 
        self.n = -3.0 
        self.Vcruise = -50.0 
        self.Vv = -20.0 
        self.Vclimb = -40.0
        self.dgr = -402
        self.rangeWS = -500
        self.stallWS = -400
        
        # self.TWs corresponds to [sustained turn, cruise, climb, takeoff, landing] for now
        
    def density(self, new_rho):
        '''Change air density (kg/m3)'''
        self.rho = new_rho
        
    def WSrange(self, WSrange):
        '''WSrange is a 1xn array with the possible wing loading values (kg/m2)'''
        self.WS = WSrange
    
    def TW_susturn(self, n, Vturn):
        '''
        Vturn:  turn speed estimate in m/s
        n:      load factor
        '''
        self.n = n
        self.Vturn = Vturn
        q = 0.5*self.rho*(Vturn**2)
        TW_susturn = q*self.CD0/(self.WS) + self.WS*self.k*(n**2)/q
        PW = (TW_susturn/self.eta_p)*Vturn
        self.TWs[0] = PW #TW_susturn
        # bonus not implemented yet: T/W >= 2*n*sqrt(CD0/(pi*AR*e))
        self.indepturnreq = 2*n*np.sqrt(self.CD0*self.k)
    
    def TW_cruise(self, Vcruise):
        '''
        Vcruise: cruise speed estimate in m/s
        '''
        self.Vcruise = Vcruise
        q = 0.5*self.rho*(Vcruise**2)
        TW_cruise = q*self.CD0*(1/self.WS) + self.k*(1/q)*self.WS
        PW = (TW_cruise/self.eta_p)*Vcruise
        self.TWs[1] = PW #TW_cruise
        
    def TW_climb(self, Vv, Vclimb):
        '''
        Vv:     climb requirement in m/s
        Vclimb: climb speed in m/s
        '''
        self.Vv = Vv
        self.Vclimb = Vclimb
        q = 0.5*self.rho*(Vclimb**2)
        TW_climb = np.ones(self.WS.size)*Vv/Vclimb + (q/self.WS)*self.CD0 + self.k*(1/q)*self.WS
        PW = (TW_climb/self.eta_p)*Vclimb
        self.TWs[2] = PW #TW_climb
        
    def TW_takeoff(self, dgr, takeoff_surface, CLto, CDto, CLmax):
        '''
        dgr:    maximum ground roll distance in m
        takeoff_surface: one of dry concrete, wet concrete, icy concrete, 
                                hard turf, firm dirt, soft turf, wet grass
                         determines friction coef
        CLto:   takeoff lift coefficient (i.e. before rotation with high lift devices deployed)
        CDto:   takeoff drag coefficient (i.e. before rotation with high lift devices deployed)
        
        '''
        if takeoff_surface == 'dry concrete':
            self.mufric = 0.04
        elif takeoff_surface == 'wet concrete':
            self.mufric = 0.05
        elif takeoff_surface == 'icy concrete':
            self.mufric = 0.02
        elif takeoff_surface == 'hard turf':
            self.mufric = 0.05
        elif takeoff_surface == 'firm dirt':
            self.mufric = 0.04
        elif takeoff_surface == 'soft turf':
            self.mufric = 0.07
        elif takeoff_surface == 'wet grass':
            self.mufric = 0.08
        else:
            print('\nTakeoff surface not recognized\nOptions are: dry concrete, wet concrete, icy concrete\n\t\t\thard turf, firm dirt, soft turf, wet grass')

        Vstall = np.sqrt((2*self.WS)/(self.rho*CLmax))
        vlof = 1.1*Vstall
        self.dgr = dgr

        q = 0.5*self.rho*(vlof**2)
        
        A = (vlof**2)/(2*self.g*dgr)
        C = q*CDto/self.WS
        D = self.mufric*(1-(q*CLto/WS))

        TWto = A + C + D
        
        TWto = (TWto/self.eta_p)*vlof*np.sqrt(2)
        self.TWs[3] = TWto

        

    def TW_landing(self, etc):
        '''Other landing requirements (TO DO in the next project when landing matters)'''
        
    def WSstall(self, Vstall, CLmax):
        '''
        Vstall: m/s
        CLmax: maximum lift coefficient (with high lift devices)
        '''
        self.Vstall = Vstall #np.sqrt((2*self.WS)/(self.rho*CLmax))
        self.CLmax = CLmax
        self.stallWS = 0.5*self.rho*(self.Vstall**2)*CLmax
        if self.stallWS > self.WS.max():
            self.WS = np.linspace(self.WS.min(), self.stallWS, self.WS.size)

        self.stallreq = True
        
    def WSmaxproprange(self, Vcruise):
        '''
        Vcruise: m/s
        '''
        q = 0.5*self.rho*(Vcruise**2)
        self.rangeWS = q*np.sqrt((1/self.k)*self.CD0)
        if self.rangeWS > self.WS.max():
            self.WS = np.linspace(self.WS.min(), self.rangeWS, self.WS.size)
        self.rangereq = True

    def plot(self, title = None, save = False):
        '''NOTE: TW is still used but all of these are converted to P/W in Watt/kg'''
        fig, ax = plt.subplots(figsize = (6, 4), dpi = 1000)
        
        TWnames = [f'Sustained Turn: n = {self.n}, {self.Vturn} m/s', f'Cruise Speed: {self.Vcruise} m/s', f'Climb Rate: {self.Vv} m/s, {self.Vclimb} m/s', f'Ground Roll: {self.dgr} m', 'Landing (LATER)']
        
        for i, TW in enumerate(self.TWs):
            if type(TW) != bool:
                # convert T/W to P/W                
                ax.plot(self.WS, TW, label = TWnames[i], path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
        
        # background weight contours
        TWmin = 10
        TWmax = 0
        for TW in self.TWs:
            if type(TW) != bool:
                specTWmin = TW.min()
                specTWmax = TW.max()
                if specTWmin < TWmin:
                    TWmin = specTWmin
                if specTWmax > TWmax:
                    TWmax = specTWmax
                    
        if type(self.indepturnreq) != bool:
            if self.indepturnreq < TWmin:
                TWmin = self.indepturnreq
            ax.plot(self.WS, self.indepturnreq*np.ones(self.WS.size), label = 'Base T/W turn req', path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])

        TWgrid = np.linspace(TWmin, TWmax, self.WS.size)
        if self.stallreq:
            ax.plot(self.stallWS*np.ones(TWgrid.size), TWgrid, label = f'Stall: {self.Vstall} m/s', path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
        
        if self.rangereq:
            ax.plot(self.rangeWS*np.ones(TWgrid.size), TWgrid, label = 'W/S of max range', path_effects = [patheffects.withTickedStroke(spacing = 10, angle = -135, length = 0.5)])
        
        # x, y = np.meshgrid(self.WS, TWgrid)
        # W0 = x*0.83612736 # 9 ft2 wing area to test
        # img = ax.contourf(x, y, W0)
        # plt.colorbar(img)
        # WORK ON THIS (raymer pg 758)

        plt.minorticks_on()
        plt.grid(True)#, which='both')
        plt.xlabel(r'W/S (kg/m$^2$)')
        plt.ylabel('P/W (watt/kg)')
        if title != None:
            plt.title(title)
        plt.legend(fontsize = 8)
        if save and title != None:
            plt.savefig(title, dpi = 1000)
        if save and title == None:
            print('Title not defined')
        plt.show()
    
    def findoptimum(self, con1, con2):
        '''con (constraint) can refer to: sustained turn, cruise speed, 
                                            climb rate, ground roll, 
                                            stall, or max range'''
        conlist = {'sustained turn':self.TWs[0], 'cruise speed':self.TWs[1], 'climb rate':self.TWs[2], 'ground roll':self.TWs[3], 'stall':self.stallWS, 'max range':self.rangeWS}
        try:
            if type(conlist[con1]) == bool:
                print(f'Constraint undetermined, please perform {con1} calculation')
                return
            elif type(conlist[con2]) == bool:
                print(f'Constraint undetermined, please perform {con1} calculation')
                return
        except:
            ##### UPDATE AS YOU ADD MORE #####
            print('\nConstraint not recognized; options are currently sustained turn, cruise speed, climb rate, ground roll, stall, or max range')
            return
        
        # find intersection between conlist
        WSlims = ['stall', 'max range']
        if con1 in WSlims and con2 in WSlims:
            print('Please select intersecting constraints')
        elif con1 in WSlims:
            TW2 = conlist[con2]
            index = np.argmin(np.abs(conlist[con1]*np.ones(TW2.size) - self.WS))
            print(f'Optimum at {TW2[index]:.4f} watt/kg and {conlist[con1]:.4f} kg/m2')
        elif con2 in WSlims:
            TW1 = conlist[con1]
            index = np.argmin(np.abs(conlist[con2]*np.ones(TW1.size) - self.WS))
            print(np.abs(conlist[con2]*np.ones(TW1.size) - TW1))
            print(index)
            print(f'Optimum at {TW1[index]:.4f} watt/kg and {conlist[con2]:.4f} kg/m2')
        else:
            TW1 = conlist[con1]
            TW2 = conlist[con2]
            index = np.argmin(np.abs(TW1-TW2))
            midanswer = (TW1[index]+TW2[index])/2
            print(f'Optimum at {midanswer:.4f} watt/kg and {self.WS[index]:.4f} kg/m2')


        
#%% using for dash 1
AR = 8      # randomly chosen
e = 0.8     # who knows if it's acurate
CD0 = 0.02  # total guess

analysis = TW_WS(AR, e, CD0) # start analysis

WS = np.linspace(5*lbfft2_kgm2, 15*lbfft2_kgm2, 100)
analysis.WSrange(WS) # initialize W/S boundaries

# put these first so they can change the W/S boundaries to match
analysis.WSstall(9, 1.1)
# analysis.WSmaxproprange(35)

analysis.TW_susturn(2, 20)
analysis.TW_cruise(25)
analysis.TW_climb(1, 20)
# analysis.TW_takeoff(100, 'dry concrete', 1.6*0.7, 0.2, 1.6)
# analysis.plot(title = 'V0 for SN-1FWHL', save = True)

analysis.findoptimum('stall', 'climb rate')


#%%
205/lbfft2_kgm2

#%%
6*lbkg

#%%
2.72/54.8
#%%
0.049635/ftm/ftm
#%%
np.sqrt(8*0.5343)

#%%
2.72*3.765