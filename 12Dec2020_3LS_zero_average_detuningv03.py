# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:32:49 2019

Resonantly excited two-level system (TLS) in the rotating wave approximation.
This script calculates several models to compare to our common measurements:
1) Time evolution starting in ground state
2) Excitation spectrum
3) Emission spectrum
4) Second-order correlation function

@author: Raju KC
"""

import matplotlib.pyplot as plt
from matplotlib import ticker # Lets the contour plot do log color scale
import numpy as np
from qutip import *
#%matplotlib inline

# The following fixes an error related to steadystate(). Solution found here:
# https://groups.google.com/forum/#!topic/qutip/HAwLlOnHoac
qutip.settings.has_mkl=False # 

# Quantum dot parameters
d0 = 1
dipole_moment1 = d0 * np.array([1, 0, 0]);    # x-polarized
dipole_moment2 = d0 * np.array([0, 1, 0]);    # y-polarized

# Field parameters
E0_array = 2*np.pi * np.array([0.1,2,3,4,5,6,7])
#for shifts,d in zip(shifts,range(len(Delta_array))):
plt.figure(figsize=(8,8))   # Gives the flexibility to change the size of the figure
labels = ['0.1*Gamma','1*Gamma','2*Gamma','3*Gamma','4*Gamma','5*Gamma','6*Gamma']

for i, E0 in enumerate(E0_array):
    #E0 = 2*np.pi * j;            # Electric field magnitude
    Epol = np.array([1, 1, 0]);  # 45degree-polarized excitation electric field of the laser light
    Efield = E0 * Epol

# Hamiltonian parameters
    Delta = 2 * np.pi * 0                      #average detuning  
    delta = 2 * np.pi * 6                      #finestructure splitting particular value in the three levels v system. can be chosen any values
    Omega1 = np.dot(dipole_moment1, Efield)    # Rabi frequency for transition from |1> to |0>
    Omega2 = np.dot(dipole_moment2, Efield)    # Rabi frequency for transition from |2> to |0>
    Gamma1 = 2 * np.pi * 1                  # Spontaneous emission rate [2*pi*GHz] for transition from |1> to |0>
    Gamma2 = 2 * np.pi * 1                 # Spontaneous emission rate [2*pi*GHz] for transition from |2> to |0>

# Operators 
    ket0 = basis(3,0)                          #(1,0,0) state vector for three levels V system
    ket1 = basis(3,1)                          #(0,1,0) state vector for three levels V system
    ket2 = basis(3,2)                          #(0,0,1) state vector for three levels V system
    a0 = ket0 * ket0.dag()                     #|ket0><ket0| 
    a1 = ket0 * ket1.dag()                     #|ket0><ket1|
    a2 = ket0 * ket2.dag()                     #|ket0><ket2|
    spont_emis1 = np.sqrt(Gamma1) * a1         # Spontaneous emission operator for |1> to |0> transition
    spont_emis2 = np.sqrt(Gamma2) * a2         # Spontaneous emission operator for |2> to |0> transition
    spont_emis = spont_emis1 + spont_emis2
    inversion = a1.dag() * a1 + a2.dag() * a2 - (a1*a1.dag())           #The "inversion" of the state is the difference
#between the excited state population and the ground state population. Here, the ground state population is 
#2*(a1*a1.dag()
    population = a1.dag() * a1 + a2.dag() * a2                              #The population of the excited state state


#----------- EMISSION SPECTRUM ------------------------------------------------

    Hrot = -0.5 * Delta * (a1*a1.dag()) + 0.5 *(Delta - delta)* (a1.dag() * a1) +0.5 * (Delta + delta) * (a2.dag() * a2)\
     + 0.5 * Omega1 * (a1.dag()+a1) + 0.5 * Omega2 * (a2.dag()+a2)

    wlist = np.linspace(-20, 20, 801) * 2 * np.pi
    spec = spectrum(Hrot, wlist, [spont_emis1,spont_emis2],(a1.dag()+a2.dag()),(a1+a2)) # Here [spont_emis1,spont_emsi2] is a list of collapsed operator (c_ops) and ai is a_op a2 is b_op
    #fig, ax = plt.subplots()
    plt.semilogy(wlist / (2 * np.pi), spec*10**(i-1))
    plt.xlabel('Emission frequency (GHz)')
    plt.ylabel('Emission intensity')
    plt.title('Emission spectrum of 3LS excited FSS/(2pi) = 6 GHz')
    plt.legend(labels,loc='best')
    
xposition = [-5,5]
for xp in xposition:
    plt.axvline(x=xp, color='r', linestyle='--') 
plt.show()
