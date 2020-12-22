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
%matplotlib inline

# The following fixes an error related to steadystate(). Solution found here:
# https://groups.google.com/forum/#!topic/qutip/HAwLlOnHoac
qutip.settings.has_mkl=False # 

# Quantum dot parameters
d0 = 1
dipole_moment1 = d0 * np.array([1, 0, 0]);    # x-polarized
dipole_moment2 = d0 * np.array([0, 1, 0]);    # y-polarized

# Field parameters
E0 = 2*np.pi * 2.5;          # Electric field magnitude
Epol = np.array([1, 1, 0]);  # 45degree-polarized excitation electric field of the laser light
Efield = E0 * Epol

# Hamiltonian parameters
#legarray = [1,2,3]
Delta_array = [2 * np.pi * 0.5,2 * np.pi * 2,2 * np.pi * 3 ,2 * np.pi * 3.5,2 * np.pi * 4,2 * np.pi * 4.5,2 * np.pi * 5,2 * np.pi * 5.5]          #average detuning  
delta = 40                                 #finestructure splitting particular value in the three levels v system. can be chosen any values
Omega1 = np.dot(dipole_moment1, Efield)    # Rabi frequency for transition from |1> to |0>
Omega2 = np.dot(dipole_moment2, Efield)    # Rabi frequency for transition from |2> to |0>
Gamma1 = 2 * np.pi * 1                     # Spontaneous emission rate [2*pi*GHz] for transition from |1> to |0>
Gamma2 = 2 * np.pi * 1                     # Spontaneous emission rate [2*pi*GHz] for transition from |2> to |0>

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

# Hamiltonian in rotating frame after rotating wave approximation (RWA)
#Hrot = -0.5 * Delta * (a1*a1.dag()) + 0.5 *(Delta -delta)* (a1.dag() * a1) +0.5 * (Delta + delta) * (a2.dag() * a2)\
     #+ 0.5 * Omega1 * (a1.dag()+a1) + 0.5 * Omega2 * (a2.dag()+a2) # Delta is average detuning and delta (small delta is FSS)

psi0 = basis(3,0)              # The initial state of the system (1,0,0) in vector notation
labels = ['abc','def','ghi']   # This label is not correct correct it later
plt.figure(figsize=(12,15))   # Gives the flexibility to change the size of the figure

# for loop to create the multiple emission spectra (Mollow septuplets)

for i,d in enumerate(range(len(Delta_array))):      # Enumaerate the for loop with indexing the each iteration 
    Hrot = -0.5 * Delta_array[d] * (a1*a1.dag()) + 0.5 *(Delta_array[d] - delta)* (a1.dag() * a1) +0.5 * (Delta_array[d] + delta) * (a2.dag() * a2)\
     + 0.5 * Omega1 * (a1.dag()+a1) + 0.5 * Omega2 * (a2.dag()+a2)   # The total Hamiltonian for a three levels V system
    wlist = np.linspace(-10, 10, 401) * 2 * np.pi     # frequency x- axid 
    spec = spectrum(Hrot, wlist, [spont_emis1,spont_emis2],(a1.dag()+a2.dag()),(a1+a2)) # Here [spont_emis1,spont_emsi2] is a list of collapsed operator (c_ops) and ai is a_op a2 is b_op
    #fig, ax = plt.subplots()
    plt.plot(wlist / (2 * np.pi), (np.log10(spec))*4 - 5*i,linewidth=2)
    plt.legend(labels[i])     # This is also not correct I need to make it again
    plt.xlabel('Emission frequency (GHz)')
    plt.ylabel('Emission intensity')
    plt.title('Emission spectrum of 3LS')
plt.show()
