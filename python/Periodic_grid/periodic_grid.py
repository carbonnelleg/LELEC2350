# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scikit-rf",
#     "scipy",
# ]
# ///
"""
Created on Wed Apr 11 10:23:01 2025

@author: carbonnelleg
"""
import skrf as rf
import scipy as sc
from scipy.constants import epsilon_0, mu_0
import numpy as np
from matplotlib import pyplot as plt

"""
_______________________________________________________________________________
Declaring variables
"""
N = 8

x_step = 0.001      # [cm]
y_step = 0.001      # [cm]

a_x = a_y = 2.417   # [cm]
w_x = 1.05          # [cm]
w_y = .242          # [cm]
delta_x = w_x/(N+1) # [cm]

x = np.arange(-w_x/2, w_x/2+x_step, x_step)
y = np.arange(-w_y/2, w_y/2+y_step, y_step)
Y, X = np.meshgrid(y, x)

freqs = sc.io.loadmat(__file__ + '/../freq.mat').get('freq').reshape(-1,)   # shape=(40,), dtype=float64
freqs *= 1e+9
Z_mom = sc.io.loadmat(__file__ + '/../Z_mom_tot.mat').get('Z_mom_tot')      # shape=(40, N, N), dtype=complex128

F_b = np.stack([np.zeros_like(X) for _ in range(N)], axis=0)                # shape=(N, x.size, y.size), dtype=float64

eta = np.sqrt(mu_0/epsilon_0)
H_inc = 1/eta
"""
End of variable declaration
_______________________________________________________________________________
Computation of rooftop basis functions and w vector
"""
for i, _ in enumerate(F_b):
    # integral of F_b[i] is 1 over area of unit cell
    x_m = (i+1)*delta_x-w_x/2
    F_b[i] = np.maximum(0., (-np.abs(X-x_m)/delta_x+1)/(delta_x*w_y))

w = H_inc*np.ones((freqs.size, N, 1))               # shape=(40, N, 1), dtype=float64
"""
_______________________________________________________________________________
Computation of x vector (coefficients of rooftop basis functions)
"""
x_coeff = np.linalg.solve(Z_mom, w)                 # shape=(40, N, 1), dtype=complex128
x_coeff *= eta**2/2
"""
_______________________________________________________________________________
Computation of transmittance (given unit incoming electric field)
"""
T = 1/(a_x*a_y  ) * x_coeff.sum(axis=1).flatten()     # shape=(40,), dtype=complex128

ntw_grid = rf.Network(__file__ + '/../withgrid.s2p')
ntw_nogrid = rf.Network(__file__ + '/../withoutgrid.s2p')

plt.plot(freqs, np.abs(T), label='Simulation')
meas_T = np.abs(ntw_grid.s[:,0,1]/ntw_nogrid.s[:,0,1])
plt.plot(ntw_grid.f, meas_T, label='Measurements')
plt.yscale('log')
plt.legend()
plt.show()
