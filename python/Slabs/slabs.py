# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:36:18 2025

@author: carbonnelleg
"""
from typing import Iterable
import numpy as np
from scipy.constants import epsilon_0, mu_0
from scipy import signal as sg
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

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
t_max = 1.6e-7
t_step = 1.0e-10
t = np.arange(t_min, t_max, t_step)
t_indices = np.arange(t.size)

f_s = 1/t_step
f_step = f_s/t.size

freqs = np.fft.ifftshift(np.fft.fftfreq(t.size, d=t_step))

# (Un)Comment these lines to change spectrum
f_0 = 3.0e9
sigma_f_0 = 0.3e9
spect = sg.windows.gaussian(t.size, std=sigma_f_0/f_step)                         
spect = np.convolve(f_step/(2*sigma_f_0*np.sqrt(2*np.pi)) *
                    (np.abs(np.abs(freqs) - f_0) < f_step/2), spect, mode='same')

# (Un)Comment these lines to change spectrum
# f_0 = 1.0e9
# spect = 1/2 * (np.abs(np.abs(freqs) - f_0) < f_step/2)

eps_air = epsilon_0
eps_rel = 4.
eps_slab = eps_rel * epsilon_0

eta = np.sqrt(mu_0/eps_air)
eta_prime = eta / np.sqrt(eps_rel)

rho = (eta_prime - eta) / (eta_prime + eta)
tau = 1 + rho
rho_prime = -rho
tau_prime = 1 + rho_prime
"""
End of variable declaration
_______________________________________________________________________________
"""


def update_slab(d1, d2, d3, d4):
    """
    Parameters
    ----------
    d1 : float
        Slab 1 start position
    d2 : float
        Slab 1 end position
    d3 : float
        Slab 2 start position
    d4 : float
        Slab 2 end position

    Returns
    -------
    is_slab : np.ndarray[bool], shape = (z.size,)
        True where z is inside a slab, function of (z)
    E_r : np.ndarray[np.complex128], shape = (z.size, t.size)
        Electric field of the wave going to the right, function of (z, t)
    E_l : np.ndarray[np.complex128]
        Electric field of the wave going to the left, function of (z, t)
    """
    # Define slab region, function of (z)
    is_slab = ((z > d1) & (z < d2)) | ((z > d3) & (z < d4))

    # Define local space variables for 5 consecutive layers
    z_local = [
        z[(z <= d1).nonzero()],
        z[((z > d1) & (z < d2)).nonzero()] - d1,
        z[((z >= d2) & (z <= d3)).nonzero()] - d2,
        z[((z > d3) & (z < d4)).nonzero()] - d3,
        z[(z >= d4).nonzero()] - d4
    ]

    # Define wavenumber in both media, function of (freqs)
    k_air = 2*np.pi*freqs*np.sqrt(eps_air * mu_0)
    k_slab = 2*np.pi*freqs*np.sqrt(eps_slab * mu_0)

    # Calculate Gamma at z=0, function of (freqs)
    Gamma = np.zeros_like(freqs, dtype=np.complex128)
    for (k, l, r) in zip(
            [k_slab, k_air, k_slab, k_air],
            [d4 - d3, d3 - d2, d2 - d1, d1],
            [rho_prime, rho, rho_prime, rho]):

        Gamma = np.exp(-2j*k*l) * (r + Gamma) / (1 + r*Gamma)

    # Calculate 5 forward (A) and backward (B) fields at the start of each z_local,
    # function of (freqs)
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

    # Calculate waves to the right (E_r) and to the left (E_l) using wave function,
    # function of (freqs, z_local)
    E_r = []
    E_l = []
    for i, k in enumerate([k_air, k_slab, k_air, k_slab, k_air]):

        E_r.append(A[i] * np.exp(-1j*np.outer(k, z_local[i])).T)
        E_l.append(B[i] * np.exp(1j*np.outer(k, z_local[i])).T)

    # Concatenate consecutive waves and perform IFFT
    E_r = np.concatenate(E_r, axis=0)
    E_r = np.fft.ifft(np.fft.ifftshift(E_r * spect, axes=1), norm='forward')
    E_l = np.concatenate(E_l, axis=0)
    E_l = np.fft.ifft(np.fft.ifftshift(E_l * spect, axes=1), norm='forward')

    return is_slab, E_r, E_l


fig1, ax = plt.subplots(num='Spectrum')
ax.stem(2*np.pi*freqs, spect,
        label=r'$\hat{E}_{0+} (z=0, \omega)$', markerfmt='b.', linefmt='blue')
ax.set_xlabel(r'$\omega$ [rad/s]')
ax.set_xticks(2*np.pi*np.array([0, f_0, -f_0]),
              labels=[0, r'$\omega_0$', r'$-\omega_0$'])
ax.legend()
fig1.suptitle('Frequency domain')

is_slab, E_r, E_l = update_slab(
    slab1_start, slab1_end, slab2_start, slab2_end)

fig2, ax = plt.subplots(num='Waves through slabs')
fig2.subplots_adjust(bottom=0.2)
slab_poly = ax.fill_between(z, -1., y2=1., where=is_slab, facecolor='lightgrey',
                            alpha=.5, label=r'Slab ($\varepsilon_r$ = 'f'{eps_rel:.2f})')
E_r_line, = ax.plot([], [], color='blue', label=r'$E_+$')
E_l_line, = ax.plot([], [], color='red', label=r'$E_-$')

ax_slider = fig2.add_axes([0.2, 0.05, 0.65, 0.03])
slider = plt.Slider(ax_slider, label='Slab 2 Start Pos',
                    valmin=slab1_end + z_step,
                    valmax=z_max - slab2_end + slab2_start,
                    valinit=slab2_start)


def update_slider(val):
    new_slab2_start = val
    new_slab2_end = new_slab2_start + slab2_end - slab2_start
    global E_r, E_l
    is_slab, E_r, E_l = update_slab(
        slab1_start, slab1_end, new_slab2_start, new_slab2_end)
    slab_poly.set_data(z, -1., 1., where=is_slab)
    fig2.canvas.draw_idle()


def init_fig() -> Iterable[plt.Artist]:
    ax.set_xlabel('Distance [m]')
    ax.set_xlim((z_min, z_max))
    ax.set_ylim((-1.1, 1.1))
    return ax.get_lines()


def update_fig(t_i) -> Iterable[plt.Artist]:
    E_r_line.set_data(z, E_r[:, t_i].real)
    E_l_line.set_data(z, E_l[:, t_i].real)
    ax.legend(loc=1)
    fig2.suptitle(f't = {t[t_i]:.2e} s')
    return E_r_line, E_l_line


slider.on_changed(update_slider)
anim = FuncAnimation(fig2, update_fig, frames=t_indices,
                     init_func=init_fig, interval=10)
plt.show()
