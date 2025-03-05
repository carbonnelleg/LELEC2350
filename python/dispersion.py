# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:06:49 2025

@author: carbonnelleg
"""

from typing import Iterable
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib import pyplot as plt
from scipy.constants import speed_of_light as c
from scipy import signal as sg

z_min = 0.
z_max = 10.
z_step = 1.0e-3
z = np.arange(z_min, z_max, z_step)

t_min = 0.
t_max = 7.0e-8
t_step = 1.0e-10
t = np.arange(t_min, t_max, t_step)
t_indices = np.arange(t.size)

f_s = 1/t_step
f_step = f_s/t.size

f_0 = 3.0e9
sigma_f_0 = 0.3e9
f_c = 2.0e9

freqs = np.fft.ifftshift(np.fft.fftfreq(t.size, d=t_step))
spect = sg.windows.gaussian(t.size, std=sigma_f_0/f_step)
spect = np.convolve(f_step/(2*sigma_f_0*np.sqrt(2*np.pi)) *
                    (np.abs(np.abs(freqs) - f_0) < f_step/2), spect, mode='same')

# Non dispersive wave
k_1 = 2*np.pi*freqs/c

H_1 = np.exp(-1j*np.outer(z, k_1))
E_1 = np.fft.ifft(np.fft.fftshift(H_1*spect, axes=1), norm='forward')

# Propagation in a waveguide with cut-off frequency f_c
k_low = np.array(np.abs(freqs) < f_c, dtype=complex)
k_low *= -2j*np.pi*np.sqrt(np.abs(f_c**2-freqs**2))/c
k_hi = np.array(np.abs(freqs) >= f_c, dtype=complex)
k_hi *= 2*np.pi*np.sign(freqs)*np.sqrt(np.abs(freqs**2-f_c**2))/c
k_2 = k_low + k_hi

H_2 = np.exp(-1j*np.outer(z, k_2))
E_2 = np.fft.ifft(np.fft.fftshift(H_2*spect, axes=1), norm='forward')


fig1, ax = plt.subplots(num='Spectrum')
ax.stem(2*np.pi*freqs, spect,
        label=r'$\hat{E}_0 (z=0, \omega)$', markerfmt='b.', linefmt='blue')
ax.set_xlabel(r'$\omega$ [rad/s]')
ax.set_xticks(2*np.pi*np.array([0, f_0, -f_0, f_c, -f_c]),
              labels=[0, r'$\omega_0$', r'$-\omega_0$', r'$\omega_c$', r'$-\omega_c$'])
ax.legend()
fig1.suptitle('Frequency domain')


fig2, ax = plt.subplots(num='Dispersion relation')
ax.plot(np.real(k_1[len(k_1)//2:]),
        2*np.pi*freqs[len(freqs)//2:],
        color='blue', label='Non-dispersive wave')
ax.plot(np.real(k_2[len(k_2)//2:]),
        2*np.pi*freqs[len(freqs)//2:],
        color='red', label=r'Dispersive wave ($\beta$)')
ax.plot(-np.imag(k_2[len(k_2)//2:]),
        2*np.pi*freqs[len(freqs)//2:],
        color='red', linestyle=':', label=r'Dispersive wave ($\alpha$)')
ax.set_xlabel(r'$\alpha,\; \beta$ [1/m]')
ax.set_ylabel(r'$\omega$ [rad/s]')
ax.set_yticks(2*np.pi*np.array([0, f_c, f_0]),
              labels=[0, r'$\omega_c$', r'$\omega_0$'])
ax.legend()
ax.grid(True, axis='y')
fig2.suptitle('Phase and group velocities')


fig3, ax = plt.subplots(num='Animation')
ln1, = ax.plot([], [], color='blue', label='Non-dispersive wave')
ln2, = ax.plot([], [], color='red', label='Dispersive wave')


def update(t_i) -> Iterable[Artist]:
    ln1.set_data(z, E_1[:, t_i].real)
    ln2.set_data(z, E_2[:, t_i].real)
    fig3.suptitle(f't = {t[t_i]:.2e} s')
    return ln1, ln2


def init_fig() -> Iterable[Artist]:
    ax.set_xlabel('Distance [m]')
    ax.legend()
    ax.set_xlim((z_min, z_max))
    ax.set_ylim((-1., 1.))
    return ln1, ln2


anim = FuncAnimation(fig3, update, frames=t_indices,
                     init_func=init_fig, interval=10)

plt.show()
