# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:36:18 2025

@author: carbonnelleg
"""

import numpy as np
from scipy.constants import epsilon_0, mu_0
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import Iterable
from matplotlib.artist import Artist
from matplotlib.animation import FuncAnimation
from scipy import signal as sg

"""
_______________________________________________________________________________
Declaring variables
"""
z_min = 0.
z_max = 10.
z_step = 4.0e-3
z = np.arange(z_min, z_max, z_step)

slab1_start, slab1_end = 2., 4.
slab2_start, slab2_end = 6., 8.

t_min = 0.
t_max = 1.5e-7
t_step = 1.0e-10
t = np.arange(t_min, t_max, t_step)
t_indices = np.arange(t.size)

f_s = 1/t_step
f_step = f_s/t.size

f_0 = 3.0e9
sigma_f_0 = 0.3e9

freqs = np.fft.ifftshift(np.fft.fftfreq(t.size, d=t_step))
spect = sg.windows.gaussian(t.size, std=sigma_f_0/f_step)
spect = np.convolve(f_step/(2*sigma_f_0*np.sqrt(2*np.pi)) *
                    (np.abs(np.abs(freqs) - f_0) < f_step/2), spect, mode='same')

eps_air = epsilon_0
eps_slab = 4 * epsilon_0
"""
End of variable declaration
_______________________________________________________________________________
"""

eta = np.sqrt(mu_0/eps_air)
eta_prime = np.sqrt(mu_0/eps_slab)

rho = (eta_prime - eta) / (eta_prime + eta)
tau = 1 + rho
rho_prime = -rho
tau_prime = 1 + rho_prime


def update_slab(d1, d2, d3, d4):
    """
    Notes
    -----

    On the simulation, it seems like wave packets are sent from the left at t != 0 such that any
    wave to the left incident to an interface will be encountering a wave to the right at
    the same moment. Those two waves will cancel out such that only a resulting wave to the
    left will be produced. It is not intended to work like that and my guess is that it has something
    to do with the fact that we are working in the frequency domain and then performing an FFT.

    Solution
    --------

    This is due to the fact that A_ is imposed, whereas A_1 should be imposed to produce the expected result.
    Now, it is only a matter of rewriting this function in matricial form where A_1=np.ones_like(freqs) and
    solving for the other forward and backward fields.
    """
    # Define air and slab region
    is_slab = ((z > d1) & (z < d2)) | \
              ((z > d3) & (z < d4))

    # Define local variables for 5 consecutive layers
    z_local = [
        z[(z <= d1).nonzero()],
        z[((z > d1) & (z < d2)).nonzero()] - d1,
        z[((z >= d2) & (z <= d3)).nonzero()] - d2,
        z[((z > d3) & (z < d4)).nonzero()] - d3,
        z[(z >= d4).nonzero()] - d4
    ]

    # Define wavenumber in both media, only dependent on freq
    k_air = 2*np.pi*freqs*np.sqrt(eps_air * mu_0)
    k_slab = 2*np.pi*freqs*np.sqrt(eps_slab * mu_0)

    # Calculate Gamma at z=0
    Gamma = np.zeros_like(freqs, dtype=np.complex128)
    for (k, l, r) in zip(
        [k_slab, k_air, k_slab, k_air],
        [d4 - d3, d3 - d2, d2 - d1, d1],
        [rho_prime, rho, rho_prime, rho]):

        Gamma = np.exp(-2j*k*l) * (r + Gamma) / (1 + r*Gamma)

    # Calculate forward (A) and backward (B) fields, only dependent on freq
    A = np.zeros((5, freqs.size), dtype=np.complex128)
    B = np.zeros((5, freqs.size), dtype=np.complex128)
    A[0] += 1.
    B[0] = A[0] * Gamma
    for i, (k, l, r, t) in enumerate(zip(
        [k_air, k_slab, k_air, k_slab],
        [d1, d2 - d1, d3 - d2, d4 - d3],
        [rho_prime, rho, rho_prime, rho],
            [tau_prime, tau, tau_prime, tau])):

        A[i+1] = 1/t * np.exp(-1j*k*l) * A[i] + r/t * np.exp(1j*k*l) * B[i]
        B[i+1] = r/t * np.exp(-1j*k*l) * A[i] + 1/t * np.exp(1j*k*l) * B[i]

    # Calculate waves to the right (E_r) and to the left (E_l) for all z using propagation
    E_r = []
    E_l = []
    for i, k in enumerate([k_air, k_slab, k_air, k_slab, k_air]):
        E_r.append(A[i] * np.exp(-1j*np.outer(k, z_local[i])).T)
        E_l.append(B[i] * np.exp(1j*np.outer(k, z_local[i])).T)

    # Concatenate both fields and perform
    E_r = np.concatenate(E_r, axis=0)
    E_r = np.fft.ifft(np.fft.ifftshift(E_r * spect, axes=1), norm='forward')
    E_l = np.concatenate(E_l, axis=0)
    E_l = np.fft.ifft(np.fft.ifftshift(E_l * spect, axes=1), norm='forward')

    return is_slab, E_r, E_l


is_slab, E_r, E_l = update_slab(
    slab1_start, slab1_end, slab2_start, slab2_end)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
slab_line, = ax.plot(z, -1+2 * is_slab, label='Slab Region')
E_r_line, = ax.plot([], [], label=r'$E_+$')
E_l_line, = ax.plot([], [], label=r'$E_-$')

ax_slider = fig.add_axes([0.2, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, label='Slab 2 Start Pos',
                valmin=slab1_end + z_step,
                valmax=z_max - slab2_end + slab2_start,
                valinit=slab2_start)


def update_slider(val):
    new_slab2_start = slider.val
    new_slab2_end = new_slab2_start + slab2_end - slab2_start
    global E_r, E_l
    new_is_slab, E_r, E_l = update_slab(
        slab1_start, slab1_end, new_slab2_start, new_slab2_end)
    slab_line.set_ydata(-1+2*new_is_slab)
    fig.canvas.draw_idle()


def init_fig():
    ax.set_xlabel('Distance [m]')
    ax.set_xlim((z_min, z_max))
    ax.set_ylim((-1.1, 1.1))
    return ax.get_lines()


def update_fig(t_i) -> Iterable[Artist]:
    E_r_line.set_data(z, E_r[:, t_i].real)
    E_l_line.set_data(z, E_l[:, t_i].real)
    ax.legend(loc=1)
    fig.suptitle(f't = {t[t_i]:.2e} s')
    return E_r_line, E_l_line


slider.on_changed(update_slider)
anim = FuncAnimation(fig, update_fig, frames=t_indices,
                     init_func=init_fig, interval=10)
plt.show()
