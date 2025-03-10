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

slab1_start, slab1_end = 2.5, 4.
slab2_start, slab2_end = 6., 8.

t_min = 0.
t_max = 1.0e-7
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


def update_slab(slab1_start, slab1_end, slab2_start, slab2_end):
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
    is_slab = ((z > slab1_start) & (z < slab1_end)) | \
              ((z > slab2_start) & (z < slab2_end))

    # Define local variables for 5 consecutive layers
    z_1 = z[(z <= slab1_start).nonzero()]
    z_2 = z[((z > slab1_start) & (z < slab1_end)).nonzero()] - slab1_start
    z_3 = z[((z >= slab1_end) & (z <= slab2_start)).nonzero()] - slab1_end
    z_4 = z[((z > slab2_start) & (z < slab2_end)).nonzero()] - slab2_start
    z_5 = z[(z >= slab2_end).nonzero()] - slab2_end

    # Define wavenumber in both media, only dependent on freq
    k_air = 2*np.pi*freqs*np.sqrt(eps_air * mu_0)
    k_slab = 2*np.pi*freqs*np.sqrt(eps_slab * mu_0)

    # Perform a time shift such that wave starts at z=0, t=0
    tot_delay = 0.
    # Add delay due to air propagation
    tot_delay += (slab1_start + slab2_start - slab1_end) * \
        np.sqrt(eps_air * mu_0)
    # Add delay due to slab propagation
    tot_delay += (slab1_end - slab1_start + slab2_end - slab2_start) * \
        np.sqrt(eps_slab * mu_0)

    # Calculate forward (A) and backward (B) fields, only dependent on freq
    # A_5 is time-delayed by tot_delay
    # A_i and B_i are shifted wrt A_{i+1} and B_{i+1}
    A_5 = np.exp(-2j*np.pi*freqs*tot_delay)
    A_4 = 1/tau_prime * A_5 * \
        np.exp(1j*k_slab*(slab2_end-slab2_start))
    B_4 = rho_prime/tau_prime * A_5 * \
        np.exp(-1j*k_slab*(slab2_end-slab2_start))
    A_3 = (1/tau * A_4 + rho/tau * B_4) * \
        np.exp(1j*k_air*(slab2_start-slab1_end))
    B_3 = (rho/tau * A_4 + 1/tau * B_4) * \
        np.exp(-1j*k_air*(slab2_start-slab1_end))
    A_2 = (1/tau_prime * A_3 + rho_prime/tau_prime * B_3) * \
        np.exp(1j*k_slab*(slab1_end-slab1_start))
    B_2 = (rho_prime/tau_prime * A_3 + 1/tau_prime * B_3) * \
        np.exp(-1j*k_slab*(slab1_end-slab1_start))
    A_1 = (1/tau * A_2 + rho/tau * B_2) * \
        np.exp(1j*k_air*slab1_start)
    B_1 = (rho/tau * A_2 + 1/tau * B_2) * \
        np.exp(-1j*k_air*slab1_start)

    # Calculate waves to the right and to the left for all z using propagation
    E_1r = A_1*np.exp(-1j*np.outer(k_air, z_1)).T
    E_1l = B_1*np.exp(1j*np.outer(k_air, z_1)).T
    E_2r = A_2*np.exp(-1j*np.outer(k_slab, z_2)).T
    E_2l = B_2*np.exp(1j*np.outer(k_slab, z_2)).T
    E_3r = A_3*np.exp(-1j*np.outer(k_air, z_3)).T
    E_3l = B_3*np.exp(1j*np.outer(k_air, z_3)).T
    E_4r = A_4*np.exp(-1j*np.outer(k_slab, z_4)).T
    E_4l = B_4*np.exp(1j*np.outer(k_slab, z_4)).T
    E_5r = A_5*np.exp(-1j*np.outer(k_air, z_5)).T
    E_5l = np.zeros_like(E_5r)
    
    # Concatenate both fields and perform IDFT
    E_r = np.concatenate((E_1r, E_2r, E_3r, E_4r, E_5r), axis=0) 
    E_r = np.fft.ifft(np.fft.ifftshift(E_r * spect, axes=1), norm='forward')
    E_r /= np.max(np.abs(E_r))
    E_l = np.concatenate((E_1l, E_2l, E_3l, E_4l, E_5l), axis=0)
    E_l = np.fft.ifft(np.fft.ifftshift(E_l * spect, axes=1), norm='forward')
    E_l /= np.max(np.abs(E_r))

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
