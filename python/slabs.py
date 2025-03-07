# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:36:18 2025

@author: carbonnelleg
"""

import numpy as np
from scipy.constants import c, epsilon_0, mu_0
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

z_min = 0.
z_max = 10.
z_step = 1.0e-3
z = np.arange(z_min, z_max, z_step)

eps_air = 1 * epsilon_0
eps_slab = 2 * epsilon_0

slab1_start, slab1_end = 2., 4.
slab2_start, slab2_end = 6., 8.

def update_slab(z, slab1, slab2_start, slab2_end):
    is_slab = ((z > slab1[0]) & (z < slab1[1])) | ((z > slab2_start) & (z < slab2_end))
    return is_slab

is_slab = update_slab(z, (slab1_start, slab1_end), slab2_start, slab2_end)
is_air = ~is_slab

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
slab_plot, = ax.plot(z, is_slab, label="Slab Region")
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("z position")
ax.legend()

ax_slider = fig.add_axes([0.2, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, label='Slab 2 Pos', valmin=4.01, valmax=7.99, valinit=6)

def update(val):
    new_slab2_start = slider.val
    new_slab2_end = new_slab2_start + 2.
    new_is_slab = update_slab(z, (slab1_start, slab1_end), new_slab2_start, new_slab2_end)
    slab_plot.set_ydata(new_is_slab)
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
