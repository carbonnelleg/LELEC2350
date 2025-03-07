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

"""
_______________________________________________________________________________
Declaring variables
"""
z_min = 0.
z_max = 10.
z_step = 1.0e-2
z = np.arange(z_min, z_max, z_step)

slab1_start, slab1_end = 2., 4.
slab2_start, slab2_end = 6., 8.

t_min = 0.
t_max = 7.0e-8
t_step = 1.0e-10
t = np.arange(t_min, t_max, t_step)
t_indices = np.arange(t.size)

f_s = 1/t_step
f_step = f_s/t.size

f_0 = 1.0e9
freqs = np.fft.ifftshift(np.fft.fftfreq(t.size, d=t_step))
spect = 1 / 2 * (np.abs(np.abs(freqs) - f_0) < f_step/2)

eps_air = epsilon_0
eps_slab = 2 * epsilon_0
"""
End of variable declaration
_______________________________________________________________________________
"""


def update_slab(z, slab1_start, slab1_end, slab2_start, slab2_end):
    is_slab = ((z > slab1_start) & (z < slab1_end)) | \
              ((z > slab2_start) & (z < slab2_end))
    eps = eps_slab*is_slab + eps_air*(~is_slab)
    k = np.outer(2*np.pi*freqs, np.sqrt(eps * mu_0))
    H = np.exp(-1j*k*z).T
    E = np.fft.ifft(np.fft.fftshift(H*spect, axes=1), norm='forward')
    return is_slab, E


is_slab, E = update_slab(z, slab1_start, slab1_end, slab2_start, slab2_end)


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
slab_line, = ax.plot(z, -1+2 * is_slab, label="Slab Region")
E_line, = ax.plot([], [])
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("Distance [m]")

ax_slider = fig.add_axes([0.2, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, label='Slab 2 Pos',
                valmin=4.01, valmax=7.99, valinit=6)


def update_slider(val):
    new_slab2_start = slider.val
    new_slab2_end = new_slab2_start + 2.
    new_is_slab, new_E = update_slab(
        z, slab1_start, slab1_end, new_slab2_start, new_slab2_end)
    slab_line.set_ydata(-1+2*new_is_slab)
    global E
    E = new_E
    fig.canvas.draw_idle()


def init_fig():
    ax.set_xlabel('Distance [m]')
    ax.legend()
    ax.set_xlim((z_min, z_max))
    ax.set_ylim((-1.1, 1.1))
    return ax.get_lines()


def update_fig(t_i) -> Iterable[Artist]:
    E_line.set_data(z, E[:, t_i].real)
    ax.legend(loc=1)
    fig.suptitle(f't = {t[t_i]:.2e} s')
    return E_line


slider.on_changed(update_slider)
anim = FuncAnimation(fig, update_fig, frames=t_indices,
                     init_func=init_fig, interval=10)
plt.show()
