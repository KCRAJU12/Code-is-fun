# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:32:49 2019

Resonantly excited two-level system (TLS) in the rotating wave approximation.
This script calculates several models to compare to our common measurements:
1) Time evolution starting in ground state
2) Excitation spectrum
3) Emission spectrum
4) Second-order correlation function

@author: Edward Flagg
"""

import matplotlib.pyplot as plt
from matplotlib import ticker # Lets the contour plot do log color scale
import numpy as np
from qutip import *
#matplotlib inline

# The following fixes an error related to steadystate(). Solution found here:
# https://groups.google.com/forum/#!topic/qutip/HAwLlOnHoac
qutip.settings.has_mkl=False # 

# Quantum dot parameters
d0 = 1
dipole_moment1 = d0 * np.array([1, 0, 0]);    # x-polarized
dipole_moment2 = d0 * np.array([0, 1, 0]);    # y-polarized

# Field parameters
E0 = 2*np.pi * 3;          # Electric field magnitude
Epol = np.array([1, 1, 0]);  # 45degree-polarized excitation electric field of the laser light
Efield = E0 * Epol

# Hamiltonian parameters
Delta = 2 * np.pi * 0                     #average detuning  
delta = 20                                  #finestructure splitting particular value in the three levels v system. can be chosen any values
Omega1 = np.dot(dipole_moment1, Efield)    # Rabi frequency for transition from |1> to |0>
Omega2 = np.dot(dipole_moment2, Efield)    # Rabi frequency for transition from |2> to |0>
Gamma1 = 2 * np.pi * 0.28                  # Spontaneous emission rate [2*pi*GHz] for transition from |1> to |0>
Gamma2 = 2 * np.pi * 0.28                  # Spontaneous emission rate [2*pi*GHz] for transition from |2> to |0>

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
Hrot = -0.5 * Delta * (a1*a1.dag()) + 0.5 *(Delta -delta)* (a1.dag() * a1) +0.5 * (Delta + delta) * (a2.dag() * a2)\
     + 0.5 * Omega1 * (a1.dag()+a1) + 0.5 * Omega2 * (a2.dag()+a2) # Delta is average detuning and delta (small delta is FSS)

psi0 = basis(3,0) # The initial state of the system

# Time-points to simulate [ns]
times = np.linspace(0, 5, 100)

result = mesolve(Hrot, psi0, times, [spont_emis], [inversion, population])

# Make a figure and plot the expectation value of the inversion
fig, ax = plt.subplots()
ax.plot(result.times, result.expect[0])
ax.plot(result.times, result.expect[1])
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Expectation Values')
ax.legend( ("Inversion", "Population"))
ax.set_title( "Time Evolution w/ Spontaneous Emission delta = 50,Delta = 2*np.pi*2" )
plt.show()

fig.savefig('Nonunitary_mesolve.pdf')

#----------- EXCITATION SPECTRUM ----------------------------------------------
# Calculated using steady-state solutions to non-unitary time evolution.

Delta_array = 2 * np.pi * np.linspace(-10, 10, 101); # [2*pi*GHz] is the average detuning
excited_pop_ss = np.zeros(len(Delta_array));
for d in range(len(Delta_array)):
    Hrot = -0.5 * Delta_array[d] * (a1*a1.dag()) + 0.5 *(Delta_array[d] -delta)* (a1.dag() * a1) +0.5 * (Delta_array[d] + delta) * (a2.dag() * a2)\
     + 0.5 * Omega1 * (a1.dag()+a1) + 0.5 * Omega2 * (a2.dag()+a2)
    final_state = steadystate(Hrot, [spont_emis]);
    # The excited state population can be found as the trace of the product of the
    # steady-state density matrix and the excited state population operator.
    excited_pop_ss[d] = (final_state * population).tr();
    #(final_state * population).tr()


fig, ax = plt.subplots()
ax.plot(Delta_array / (2 * np.pi), excited_pop_ss)
ax.set_xlabel('Average detuning, \Delta/2\pi (GHz)')
ax.set_ylabel('Excited state population')
ax.set_title('Excitation spectrum of 3LS delta = 50,Delta = 2*np.pi*2')
plt.show()

#----------- EMISSION SPECTRUM ------------------------------------------------

Hrot = -0.5 * Delta * (a1*a1.dag()) + 0.5 *(Delta - delta)* (a1.dag() * a1) +0.5 * (Delta + delta) * (a2.dag() * a2)\
     + 0.5 * Omega1 * (a1.dag()+a1) + 0.5 * Omega2 * (a2.dag()+a2)

wlist = np.linspace(-10, 10, 401) * 2 * np.pi
spec = spectrum(Hrot, wlist, [spont_emis1,spont_emis2],(a1.dag()+a2.dag()),(a1+a2)) # Here [spont_emis1,spont_emsi2] is a list of collapsed operator (c_ops) and ai is a_op a2 is b_op
fig, ax = plt.subplots()
ax.plot(wlist / (2 * np.pi), spec)
ax.set_xlabel('Emission frequency (GHz)')
ax.set_ylabel('Emission intensity')
ax.set_title('Emission spectrum of 3LS excited')
plt.show()

#----------- EMISSION SPECTRUM vs EXCITATION DETUNING -------------------------

wlist = np.linspace(-5, 5, 401) * 2 * np.pi
Delta_array = 2 * np.pi * np.linspace(-14, 14, 101);

spec = np.zeros( (len(Delta_array),len(wlist)) );
for d in range(len(Delta_array)):
    Hrot = -0.5 * Delta_array[d] * (a1*a1.dag()) + 0.5 *(Delta_array[d] -delta)* (a1.dag() * a1) +0.5 * (Delta_array[d] + delta) * (a2.dag() * a2)\
     + 0.5 * Omega1 * (a1.dag()+a1) + 0.5 * Omega2 * (a2.dag()+a2)
    # Caculate emission spectrum
    spec[d] = spectrum(Hrot, wlist, [spont_emis1,spont_emis2],(a1.dag()+a2.dag()),(a1+a2)) # Here [spont_emis1,spont_emsi2] i

fig, ax = plt.subplots()
#ax.contourf(wlist / (2*np.pi), Delta_array / (2*np.pi), spec, \
#            locator=ticker.LogLocator())
ax.contourf(wlist / (2*np.pi), Delta_array / (2*np.pi), spec)
ax.set_xlabel('Emission frequency (GHz)')
ax.set_ylabel('Excitation detuning (GHz)')
ax.set_title('Emission spectrum vs detuning')

#----------- SECOND-ORDER CORRELATION -----------------------------------------

# Hamiltonian in rotating frame after rotating wave approximation (RWA)
Hrot = -0.5 * Delta * (a1*a1.dag()) + 0.5 *(Delta - delta)* (a1.dag() * a1) +0.5 * (Delta + delta) * (a2.dag() * a2)\
     + 0.5 * Omega1 * (a1.dag()+a1) + 0.5 * Omega2 * (a2.dag()+a2)     
# Times at which to simulate
taus = np.linspace(0, 5.0, 200)

# Calculate the (un-normalized) correlation function, G2.
#G2 = correlation_3op_1t(Hrot, None, taus, [spont_emis1,spont_emis2], \
       # destroy(2).dag(), destroy(2).dag()*destroy(2), destroy()

G2 = correlation_3op_1t(Hrot, None, taus, [spont_emis1,spont_emis2], \
      , (a1.dag() + a2.dag()), (a1.dag()*a1 + a2.dag()*a2), (a1 + a1))


# Normalize the correlation by the steady-state population.
final_state = steadystate(Hrot, [spont_emis1,spont_emis2]);
g2 = G2 / (final_state * population).tr()**2;

fig, ax = plt.subplots()
ax.plot(taus, np.real(g2))
ax.set_xlabel('Time tau (ns)')
ax.set_ylabel('g(2)')
ax.set_title('Second-order correlation function')
plt.show()