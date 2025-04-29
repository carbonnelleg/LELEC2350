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
from scipy.integrate import trapezoid as trap
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

freqs = sc.io.loadmat(__file__ + '/../data/freq.mat').get('freq').reshape(-1,)   # shape=(40,), dtype=float64
freqs *= 1e+9
Z_mom = sc.io.loadmat(__file__ + '/../data/Z_mom_tot.mat').get('Z_mom_tot')      # shape=(freqs.size, N, N), dtype=complex128

F_b = np.stack([np.zeros_like(X) for _ in range(N)], axis=0)                # shape=(N, x.size, y.size), dtype=float64

eta = np.sqrt(mu_0/epsilon_0)
H_inc = 1/eta
"""
End of variable declaration
_______________________________________________________________________________
Computation of rooftop basis functions and w vector
"""
tol = 1e-3
for i, _ in enumerate(F_b):
    x_m = (i+1)*delta_x-w_x/2
    F_b[i] = np.maximum(0., (-np.abs(X-x_m)/delta_x+1)/(delta_x*w_y))
    # integral of F_b[i] should be 1.0 over area of unit cell
    I = trap(trap(F_b[i], dx=y_step), dx=x_step)
    assert np.abs(I - 1.0) < tol

w = H_inc*np.ones((freqs.size, N, 1))                   # shape=(freqs.size, N, 1), dtype=float64
"""
_______________________________________________________________________________
Computation of x vector (coefficients of rooftop basis functions)
"""
x_coeff = np.linalg.solve(Z_mom, w)                     # shape=(freqs.size, N, 1), dtype=complex128
x_coeff *= eta**2/2
"""
_______________________________________________________________________________
Computation of transmittance (given unit incoming electric field)
"""
S21_sim = 1/(a_x*a_y*1e3) * x_coeff.sum(axis=1).flatten()    # shape=(freqs.size,), dtype=complex128
"""
_______________________________________________________________________________
Create sk-rf networks of the simulation and the measurements
"""
S = np.array([[np.zeros_like(S21_sim), S21_sim], [S21_sim, np.zeros_like(S21_sim)]]).T
sim_ntw = rf.Network(frequency=freqs, s=S, name='Simulation')

grid_ntw = rf.Network(__file__ + '/../data/withgrid.s2p')
nogrid_ntw = rf.Network(__file__ + '/../data/withoutgrid.s2p')
meas_ntw = grid_ntw/nogrid_ntw
meas_ntw.name = 'Measurements'

sim_ntw.plot_s_db(m=1, n=0)
meas_ntw.plot_s_db(m=1, n=0)
grid_ntw.plot_s_db(m=1, n=0)
plt.show()
