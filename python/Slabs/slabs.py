# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "tqdm",
# ]
# ///
"""
Created on Mon Mar  3 18:36:18 2025

@author: carbonnelleg
"""
from typing import Iterable
import argparse
from tqdm import tqdm
import numpy as np
from scipy.constants import epsilon_0, mu_0
from scipy import signal as sg
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider, Button, RadioButtons

"""
_______________________________________________________________________________
Declaring variables
"""
x_min = -2.5
x_max = 2.5

z_min = 0.
z_max = 5.

# Setting steps to None will automatically choose appropriate values according to the simulation type
x_step = None
z_step = None

slab1_end = 1.
slab2_start, slab2_end = 2.5, 3.5

theta = 0. # [°]

t_min = 0.
t_step = 1.0e-10

# Setting t_max to None will automatically choose appropriate value for t_max and f_step according to the simulation type
t_max = None

single_freq_f_0 = 0.5e9
wave_packet_f_0 = 3.0e9
sigma_f_0 = 0.3e9

eps_rel = 4.
"""
End of variable declaration
_______________________________________________________________________________
"""


class Simulation:

    def __init__(
        self,
        # Space grid initialization
        x_min=x_min,
        x_max=x_max,
        z_min=z_min,
        z_max=z_max,
        x_step=x_step,
        z_step=z_step,
        # Initial positions of the slabs
        slab1_end=slab1_end,
        slab2_start=slab2_start,
        slab2_end=slab2_end,
        # Initial incident angle
        theta=theta,
        # Time grid initialization
        t_min=t_min,
        t_step=t_step,
        t_max=None,
        # Spectrum initialization
        single_freq=True,
        f_0=None,
        sigma_f_0=sigma_f_0,
        # Physical parameters
        eps_air=epsilon_0,
        eps_rel=eps_rel,
        mu_0=mu_0,
        # 1D simulation or 2D simulation
        is_2D=False,
        # Possibility of adding sliders and buttons
        sliders_and_buttons=False,
        **kwargs
    ):
        self.is_2D = is_2D
        self.single_freq = single_freq

        # Space grid initialization
        self.x_min = x_min
        self.x_max = x_max
        self.z_min = z_min
        self.z_max = z_max
        self.x_step = x_step
        self.z_step = z_step

        # Initial positions of the slabs
        self.slab1_end = slab1_end
        self.slab2_start = slab2_start
        self.slab2_end = slab2_end

        # Time grid initialization
        # Time array should preferably be of length 2**N
        self.t_min = t_min
        self.t_step = t_step
        self.t_max = t_max

        # Spectrum initialization
        self.f_s = 1/t_step
        if f_0 is None:
            if single_freq: f_0 = single_freq_f_0
            else: f_0 = wave_packet_f_0
        self.f_0 = f_0
        self.sigma_f_0 = sigma_f_0

        # Initial incident angle
        self.theta = np.deg2rad(theta)

        # Direction vectors in (z, x)
        # dependent on (theta)
        self.k_r = np.array([0. + 0.j, 0. + 0.j])
        self.k_l = np.array([0. + 0.j, 0. + 0.j])
        self.k_prime_r = np.array([0. + 0.j, 0. + 0.j])
        self.k_prime_l = np.array([0. + 0.j, 0. + 0.j])

        # Physical parameters
        self.eps_air = eps_air
        self.eps_rel = eps_rel
        self.eps_slab = eps_rel * eps_air
        self.mu_0 = mu_0
        # Impedances
        # dependent on (theta)
        self.eta = np.sqrt(mu_0/eps_air)
        self.eta_prime = self.eta / np.sqrt(eps_rel)
        self.eta_TE, self.eta_prime_TE = 0. + 0.j, 0. + 0.j
        self.eta_TM, self.eta_prime_TM = 0 + 0.j, 0. + 0.j
        # Reflexion and transmission coefficients
        # dependent on (theta)
        self.rho_TE, self.rho_prime_TE = 0. + 0.j, 0. + 0.j
        self.tau_TE, self.tau_prime_TE = 0. + 0.j, 0. + 0.j
        self.rho_TM, self.rho_prime_TM = 0. + 0.j, 0. + 0.j
        self.tau_TM, self.tau_prime_TM = 0. + 0.j, 0. + 0.j

        # Update direction vectors and reflexion and transmission coefficients for current (theta)
        self.update_theta()
        # Initialize arrays
        self._init_arrays(**kwargs)
        # Initialize matplotlib figures
        self._init_figs(**kwargs)
        # Initialize buttons and sliders
        self.sliders_and_buttons = sliders_and_buttons
        if sliders_and_buttons:
            self._init_sliders_buttons(**kwargs)

    def _init_arrays(self, **kwargs):
        # Choose appropriate x_step
        if self.x_step is None:
            if self.is_2D and not self.single_freq:
                x_step = (self.x_max-self.x_min)/250
            else:
                x_step = (self.x_max-self.x_min)/400
        else:
            x_step = self.x_step
        
        # Choose appropriate z_step
        if self.z_step is None:
            if self.is_2D and not self.single_freq:
                z_step = (self.z_max-self.z_min)/250
            else:
                z_step = (self.z_max-self.z_min)/400
        else:
            z_step = self.z_step
        
        # Choose appropriate t_max and f_step
        if self.t_max is None:
            if self.single_freq:
                t_max = self.t_min + 128*self.t_step
            else:
                if self.is_2D:
                    t_max = self.t_min + 512*self.t_step
                else:
                    t_max = self.t_min + 1024*self.t_step
        else:
            t_max = self.t_max
        self.f_step = 1 / (t_max-self.t_min)
        
        # Space grid initialization
        if self.is_2D:
            self.x, self.z = np.meshgrid(np.arange(self.x_min, self.x_max, x_step),
                                         np.arange(self.z_min, self.z_max, z_step))
            self.X, self.Z = np.meshgrid(np.arange(self.x_min, self.x_max+x_step, x_step),
                                         np.arange(self.z_min, self.z_max+z_step, z_step))
        else:
            self.z = np.arange(self.z_min, self.z_max, z_step)
        
        # Time grid initialization
        self.t = np.arange(self.t_min, t_max, self.t_step)
        self.t_indices = np.arange(self.t.size, dtype=np.uint16)

        # Spectrum initialization
        self.freqs = np.fft.ifftshift(
            np.fft.fftfreq(self.t.size, d=self.t_step))
        if self.single_freq:
            # FT of single frequency is 2 deltas at +/- f_0
            self.spect = 1/2 * \
                (np.abs(np.abs(self.freqs) - self.f_0) < self.f_step/2)
        else:
            # Gaussian window, shifted around +/- f_0, normalized
            self.spect = sg.windows.gaussian(
                self.t.size, std=self.sigma_f_0/self.f_step)
            self.spect = np.convolve(
                np.abs(np.abs(self.freqs) - self.f_0) < self.f_step/2,
                self.spect, mode='same')
            self.spect *= self.f_step/(2*self.sigma_f_0*np.sqrt(2*np.pi))
        self.spect = self.spect.astype(np.complex128)
        self.spect *= np.sqrt(2)/2 * np.exp(-1j*self.freqs*self.t_min)

        # Arrays function of (z)
        self.is_slab = self.get_is_slab()
        # Arrays function of (freqs)
        self.k_air = 2*np.pi*self.freqs*np.sqrt(self.eps_air * self.mu_0)
        self.k_slab = 2*np.pi*self.freqs*np.sqrt(self.eps_slab * self.mu_0)
        self.Gamma_TE = np.zeros_like(self.freqs, dtype=np.complex128)
        self.Gamma_TM = np.zeros_like(self.freqs, dtype=np.complex128)
        self.A_TE = np.zeros((4, self.freqs.size), dtype=np.complex128)
        self.A_TE[0] += 1.
        self.B_TE = np.zeros((4, self.freqs.size), dtype=np.complex128)
        self.A_TM = np.zeros((4, self.freqs.size), dtype=np.complex128)
        self.A_TM[0] += 1.
        self.B_TM = np.zeros((4, self.freqs.size), dtype=np.complex128)
        # Arrays function of (z, freqs)
        self.E_r_TE = np.zeros((*self.z.shape, *self.freqs.shape),
                               dtype=np.complex128)
        self.E_l_TE = np.zeros((*self.z.shape, *self.freqs.shape),
                               dtype=np.complex128)
        self.E_r_TM = np.zeros((*self.z.shape, *self.freqs.shape),
                               dtype=np.complex128)
        self.E_l_TM = np.zeros((*self.z.shape, *self.freqs.shape),
                               dtype=np.complex128)
    
    def print_array_memory_size(self):
        print('Arrays memory size:')
        print('________________________')
        print(f'{self.z.nbytes=}')
        if self.is_2D:
            print(f'{self.x.nbytes=}')
            print(f'{self.Z.nbytes=}')
            print(f'{self.X.nbytes=}')
        print(f'{self.t.nbytes=}')
        print(f'{self.t_indices.nbytes=}')
        print(f'{self.freqs.nbytes=}')
        print(f'{self.spect.nbytes=}')
        print(f'{self.is_slab.nbytes=}')
        print(f'{self.k_air.nbytes=}')
        print(f'{self.k_slab.nbytes=}')
        print(f'{self.Gamma_TE.nbytes=}')
        print(f'{self.Gamma_TM.nbytes=}')
        print(f'{self.A_TE.nbytes=}')
        print(f'{self.A_TM.nbytes=}')
        print(f'{self.B_TE.nbytes=}')
        print(f'{self.B_TM.nbytes=}')
        print(f'{self.E_r_TE.nbytes=}')
        print(f'{self.E_r_TM.nbytes=}')
        print(f'{self.E_l_TE.nbytes=}')
        print(f'{self.E_l_TM.nbytes=}')
        print('________________________')
        print('Total array memory size:')
        total_n_bytes = self.z.nbytes + self.t.nbytes + self.t_indices.nbytes + \
                        self.freqs.nbytes + self.spect.nbytes + self.is_slab.nbytes + \
                        self.k_air.nbytes + self.k_slab.nbytes + self.Gamma_TE.nbytes + \
                        self.Gamma_TM.nbytes + self.A_TE.nbytes + self.A_TM.nbytes + \
                        self.B_TE.nbytes + self.B_TM.nbytes + self.E_r_TE.nbytes +\
                        self.E_r_TM.nbytes + self.E_l_TE.nbytes + self.E_l_TM.nbytes
        if self.is_2D:
            total_n_bytes += self.x.nbytes + self.Z.nbytes + self.X.nbytes

        print(f'{total_n_bytes / 2**30:.2f} GiB')

    def _init_figs(self, **kwargs):
        # Figure of the spectrum
        # ______________________________________________________________________
        self.fig1, self.ax1 = plt.subplots(num='Spectrum')
        self.ax1.stem(2*np.pi*self.freqs, np.abs(self.spect),
                      label=r'$\left.\hat{E}_{0+}\right|_{z=0, x=0}\,(\omega)$',
                      markerfmt='b.', linefmt='blue', basefmt='blue')
        self.ax1.set_xlabel(r'$\omega$ [rad/s]')
        self.ax1.set_xticks(2*np.pi*np.array([0, self.f_0, -self.f_0]),
                            labels=[0, r'$\omega_0$', r'$-\omega_0$'])
        self.ax1.legend()
        self.ax1.set_title(fr'$f_0$ = {self.f_0:.1e}')
        self.fig1.suptitle('Frequency domain')
        # ______________________________________________________________________

        # Animated figures of the waves through slabs
        # (z, x)-dimension plot, animated over time
        # ______________________________________________________________________
        if self.is_2D:
            self.fig2, self.ax2 = plt.subplots(num='2D Waves through slabs')
            self.fig2.subplots_adjust(right=0.85)
            # Initialize collections
            vmax = 2. if self.single_freq else 1.
            self.E_TE_mesh = self.ax2.pcolormesh(self.Z, self.X, np.zeros_like(self.z), cmap='Blues',
                                                       shading='flat', alpha=.8, vmin=0., vmax=vmax)
            self.E_TM_mesh = self.ax2.pcolormesh(self.Z, self.X, np.zeros_like(self.z), cmap='Reds',
                                                       shading='flat', alpha=.8, vmin=0., vmax=vmax)
            self.E_45_mesh = self.ax2.pcolormesh(self.Z, self.X, np.zeros_like(self.z), cmap='Greens',
                                                    shading='flat', alpha=.8, vmin=0., vmax=vmax)
            for col in self.ax2.collections:
                col.set_visible(False)
            # Choose which polarization to show
            # self.E_TE_mesh.set_visible(True)
            # self.E_TM_mesh.set_visible(True)
            self.E_45_mesh.set_visible(True)
            # Add colorbar
            self.cbar = self.fig2.colorbar(self.E_45_mesh, ax=self.ax2, fraction=0.046, pad=0.04)
            self.cbar.set_label('E-field magnitude')
            # Add horizontal and vertical lines
            self.d1_vline = self.ax2.axvline(self.slab1_end, ls="--", lw=1., c="black")
            self.d2_vline = self.ax2.axvline(self.slab2_start, ls="--", lw=1., c="black")
            self.d3_vline = self.ax2.axvline(self.slab2_end, ls="--", lw=1., c="black")
            # Add arrow
            self.arrow_l = (self.z_max-self.z_min)/10
            self.arrow = self.ax2.arrow(self.z_min, (self.x_max+self.x_min)/2,
                                        dx=self.arrow_l*self.k_prime_r[0].real,
                                        dy=self.arrow_l*self.k_prime_r[1].real,
                                        width=0.05, length_includes_head=True,
                                        fc='black', ec='black')
            # Labels and limits
            self.ax2.set_xlabel('z position [m]')
            self.ax2.set_ylabel('x position [m]')
            self.ax2.set_xlim([self.z_min, self.z_max])
            self.ax2.set_ylim([self.x_min, self.x_max])
        # ______________________________________________________________________

        # z-dimension plot, animated over time
        # ______________________________________________________________________
        else:
            self.fig2, self.ax2 = plt.subplots(num='1D Waves through slabs')
            # Initialize lines
            self.slab_poly = self.ax2.fill_between(
                self.z, -2., 2., where=self.is_slab, facecolor='grey',
                alpha=.5, label=r'Slab ($\varepsilon_r$ = 'f'{self.eps_rel:.2f})')
            self.E_r_TE_line, = self.ax2.plot(self.z, np.take(self.E_r_TE, 0, axis=-1).real,
                                              color='blue')
            self.E_l_TE_line, = self.ax2.plot(self.z, np.take(self.E_r_TE, 0, axis=-1).real,
                                              color='red')
            self.E_sum_TE_line, = self.ax2.plot(self.z, np.take(self.E_r_TE, 0, axis=-1).real,
                                                color='green')
            self.E_r_TM_line, = self.ax2.plot(self.z, np.take(self.E_r_TE, 0, axis=-1).real,
                                              color='blue')
            self.E_l_TM_line, = self.ax2.plot(self.z, np.take(self.E_r_TE, 0, axis=-1).real,
                                              color='red')
            self.E_sum_TM_line, = self.ax2.plot(self.z, np.take(self.E_r_TE, 0, axis=-1).real,
                                                color='green')
            self.E_r_45_line, = self.ax2.plot(self.z, np.take(self.E_r_TE, 0, axis=-1).real,
                                              color='blue')
            self.E_l_45_line, = self.ax2.plot(self.z, np.take(self.E_r_TE, 0, axis=-1).real,
                                              color='red')
            self.E_sum_45_line, = self.ax2.plot(self.z, np.take(self.E_r_TE, 0, axis=-1).real,
                                                color='green')
            for line in self.ax2.lines:
                line.set_visible(False)
            # Choose which polarization to show
            self.E_l_line = self.E_l_45_line
            self.E_r_line = self.E_r_45_line
            self.E_sum_line = self.E_sum_45_line
            # self.E_l_line.set_visible(True)
            # self.E_r_line.set_visible(True)
            self.E_sum_line.set_visible(True)
            # Labels and limits
            self.ax2.set_xlabel('z position [m]')
            self.ax2.set_xlim([self.z_min, self.z_max])
            self.ax2.set_ylabel('E-field magnitude')
            if self.single_freq:
                self.ax2.set_ylim([-2, 2])
            else:
                self.ax2.set_ylim([-1.1, 1.1])
        # ______________________________________________________________________
        self.ax2_title = self.ax2.set_title(f't = {self.t[0]:.2e} s', y=.9,
                                            bbox=dict(facecolor='lightgrey', alpha=1.))
        # Animation parameters
        self.is_animated = False
        self.current_frame = 0
    
    def _init_sliders_buttons(self, **kwargs):
        # Adjust position of plots
        self.fig2.subplots_adjust(right=0.7, bottom=0.25)

        # Update function for slab2 starting position
        def update_slab2_slider(val):
            self.change_slab2_pos(val)
        
        # Initialize slab2 slider
        ax_slab2_slider = self.fig2.add_axes([0.2, 0.05, 0.65, 0.03])
        self.slab2_slider = Slider(
            ax_slab2_slider, label='Slab 2 Start Pos',
            valmin=self.slab1_end,
            valmax=self.z_max - self.slab2_end + self.slab2_start,
            valinit=self.slab2_start)
        self.slab2_slider.on_changed(update_slab2_slider)
        
        # Update function for theta value
        def update_theta_slider(val):
            self.set_theta(np.deg2rad(val))
            if self.is_2D:
                self.arrow.set_data(dx=self.arrow_l*self.k_prime_r[0].real,
                                    dy=self.arrow_l*self.k_prime_r[1].real)

        # Initialize theta slider
        ax_theta_slider = self.fig2.add_axes([0.2, 0.1, 0.65, 0.03])
        self.theta_slider = Slider(
            ax_theta_slider, label=r'$\theta$ [°]',
            valmin=-90., valmax=90.,
            closedmin=False, closedmax=False,
            valinit=np.rad2deg(self.theta))
        self.theta_slider.on_changed(update_theta_slider)

        # Toggle function for 1D/2D figures
        def toggle_1D_2D(event):
            self.switch_1D_2D()
            plt.show()

        # Initialize 1D/2D button
        ax_1D_2D_button = self.fig2.add_axes([0.81, 0.9, 0.18, 0.05])
        label = '1D' if self.is_2D else '2D'
        self._1D_2D_button = Button(ax_1D_2D_button, label)
        self._1D_2D_button.on_clicked(toggle_1D_2D)
        
        # Toggle function to update visibility of left-right fields
        def toggle_left_right_display(label):
            self.switch_left_right_display(label)
            self.fig2.canvas.draw_idle()

        # Initialize left-right radio button (E_r, E_l vs E_tot)
        if not self.is_2D:
            ax_left_right_radio_buttons = self.fig2.add_axes([0.81, 0.49, 0.18, 0.15])
            self.left_right_radio_buttons = RadioButtons(
                ax_left_right_radio_buttons, labels=[r'$E_+$, $E_-$', r'$E_{tot}$'],
                active=1)
            self.left_right_radio_buttons.on_clicked(toggle_left_right_display)

        # Toggle function to update visibility of TE-TM modes
        def toggle_TE_TM_display(label):
            self.switch_TE_TM_display(label)
            if not self.is_2D:
                label = self.left_right_radio_buttons.value_selected
                toggle_left_right_display(label)
            self.fig2.canvas.draw_idle()

        # Initialize TE-TM radio button (E_TE vs E_TM vs E_45°)
        ax_TE_TM_radio_buttons = self.fig2.add_axes([0.81, 0.66, 0.18, 0.15])
        self.TE_TM_radio_buttons = RadioButtons(
                ax_TE_TM_radio_buttons, labels=[r'$E_{TE}$', r'$E_{TM}$', r'$E_{45}$'],
                active=2)
        self.TE_TM_radio_buttons.on_clicked(toggle_TE_TM_display)

        # Toggle function for start/stop
        def toggle_start_stop(event):
            if self.start_stop_button.label.get_text() == 'Start':
                self.start()
            else:
                self.stop()
            self.fig2.canvas.draw_idle()
        
        # Initialize start/stop button
        ax_start_stop_button = self.fig2.add_axes([0.81, 0.83, 0.18, 0.05])
        self.start_stop_button = Button(ax_start_stop_button, 'Start')
        self.start_stop_button.on_clicked(toggle_start_stop)

        # Initialize pause/play button
        if self.is_2D:
            ax_pause_play_button = self.fig2.add_axes([0.81, 0.32, 0.18, 0.05])
        else:
            ax_pause_play_button = self.fig2.add_axes([0.81, 0.27, 0.18, 0.05])
        self.pause_play_button = Button(ax_pause_play_button, 'Pause',
                                        color='0.5', hovercolor='0.5')

        # Initialize loop back button
        if self.is_2D:
            ax_loop_back_button = self.fig2.add_axes([0.81, 0.25, 0.18, 0.05])
        else:
            ax_loop_back_button = self.fig2.add_axes([0.81, 0.2, 0.18, 0.05])
        self.loop_back_button = Button(ax_loop_back_button, 'Loop back',
                                       color='0.5', hovercolor='0.5')

    def change_slab2_pos(self, pos):
        self.slab2_end = pos + self.slab2_end - self.slab2_start
        self.slab2_start = pos
        # Update slab region, function of (z)
        self.is_slab = self.get_is_slab()
        if self.is_2D and self.d2_vline and self.d3_vline:
            self.d2_vline.set_xdata([self.slab2_start, self.slab2_start])
            self.d3_vline.set_xdata([self.slab2_end, self.slab2_end])
        elif not self.is_2D and self.slab_poly:
            self.slab_poly.set_data(self.z, -2., 2., where=self.is_slab)

    def update_theta(self):
        th_s = np.complex128(self.theta)
        th_0 = np.arcsin(np.sin(th_s) * np.sqrt(self.eps_rel))

        # Updates direction vectors in (z, x)
        self.k_r = np.array([np.cos(th_0),
                             np.sin(th_0)])
        self.k_l = np.array([-np.cos(th_0),
                             np.sin(th_0)])
        self.k_prime_r = np.array([np.cos(th_s),
                                   np.sin(th_s)])
        self.k_prime_l = np.array([-np.cos(th_s),
                                   np.sin(th_s)])
        
        # Update impedances
        self.eta_TE = self.eta/np.cos(th_0)
        self.eta_TM = self.eta*np.cos(th_0)
        self.eta_prime_TE = self.eta_prime/np.cos(th_s)
        self.eta_prime_TM = self.eta_prime*np.cos(th_s)

        # Updates reflexion and transmission coefficients
        self.rho_TE = (self.eta_prime_TE - self.eta_TE) / \
                      (self.eta_prime_TE + self.eta_TE)
        self.tau_TE = 1 + self.rho_TE
        self.rho_prime_TE = -self.rho_TE
        self.tau_prime_TE = 1 + self.rho_prime_TE

        self.rho_TM = (self.eta_prime_TM - self.eta_TM) / \
                      (self.eta_prime_TM + self.eta_TM)
        self.tau_TM = 1 + self.rho_TM
        self.rho_prime_TM = -self.rho_TM
        self.tau_prime_TM = 1 + self.rho_prime_TM

    def set_theta(self, theta):
        self.theta = theta
        self.update_theta()
        if self.arrow and self.is_2D:
            self.arrow.set_data(dx=self.arrow_l*self.k_prime_r[0].real,
                                dy=self.arrow_l*self.k_prime_r[1].real)

    def get_is_slab(self):
        return ((self.z < self.slab1_end)) | \
               ((self.z > self.slab2_start) & (self.z < self.slab2_end))
    
    def update_arrays(self, *args):
        """
        Updates the different fields and waves of the simulation for the current
        slab positions.
        """
        z = self.z
        # Positions
        d1, d2, d3 = self.slab1_end, self.slab2_start, self.slab2_end
        # Define local space variables for 5 consecutive layers
        z_local = [
            z[np.unique((z < d1).nonzero()[0])],
            z[np.unique(((z >= d1) & (z <= d2)).nonzero()[0])] - d1,
            z[np.unique(((z > d2) & (z < d3)).nonzero()[0])] - d2,
            z[np.unique((z >= d3).nonzero()[0])] - d3]

        # Wave vectors
        k_0_z = self.k_air*self.k_r[0]
        k_s_z = self.k_slab*self.k_prime_r[0]
        k_0_z = k_0_z.real - 1j*np.abs(k_0_z.imag)
        k_s_z = k_s_z.real - 1j*np.abs(k_s_z.imag)
        # Reflexion and transmission coefficients
        r_0_TE, r_s_TE = self.rho_TE, self.rho_prime_TE
        t_0_TE, t_s_TE = self.tau_TE, self.tau_prime_TE
        r_0_TM, r_s_TM = self.rho_TM, self.rho_prime_TM
        t_0_TM, t_s_TM = self.tau_TM, self.tau_prime_TM

        # Calculate Gamma at z=0 by going backwards in z-direction
        # function of (freqs)
        self.Gamma_TE *= 0
        self.Gamma_TM *= 0
        for i, (k_z, l, r_TE, r_TM) in enumerate(zip(
            [k_s_z, k_0_z, k_s_z],
            [d3 - d2, d2 - d1, d1],
            [r_s_TE, r_0_TE, r_s_TE],
            [r_s_TM, r_0_TM, r_s_TM])):

            self.Gamma_TE = np.exp(-2j*k_z*l) * (r_TE + self.Gamma_TE) / (1 + r_TE*self.Gamma_TE)
            self.Gamma_TM = np.exp(-2j*k_z*l) * (r_TM + self.Gamma_TM) / (1 + r_TM*self.Gamma_TM)

        # Calculate 4 forward (A) and backward (B) waves at the start of each z_local
        # function of (freqs)
        self.B_TE[0] = self.A_TE[0] * self.Gamma_TE
        self.B_TM[0] = self.A_TM[0] * self.Gamma_TM
        for i, (k_z, l, r_TE, r_TM, t_TE, t_TM) in enumerate(zip(
            [k_s_z, k_0_z, k_s_z],
            [d1, d2 - d1, d3 - d2],
            [r_0_TE, r_s_TE, r_0_TE],
            [r_0_TM, r_s_TM, r_0_TM],
            [t_0_TE, t_s_TE, t_0_TE],
            [t_0_TM, t_s_TM, t_0_TM])):
            
            self.A_TE[i+1] = 1/t_TE * np.exp(-1j*k_z*l) * self.A_TE[i] + \
                             r_TE/t_TE * np.exp(1j*k_z*l) * self.B_TE[i]
            self.B_TE[i+1] = r_TE/t_TE * np.exp(-1j*k_z*l) * self.A_TE[i] + \
                             1/t_TE * np.exp(1j*k_z*l) * self.B_TE[i]
            self.A_TM[i+1] = 1/t_TM * np.exp(-1j*k_z*l) * self.A_TM[i] + \
                             r_TM/t_TM * np.exp(1j*k_z*l) * self.B_TM[i]
            self.B_TM[i+1] = r_TM/t_TM * np.exp(-1j*k_z*l) * self.A_TM[i] + \
                             1/t_TM * np.exp(1j*k_z*l) * self.B_TM[i]
        self.B_TE[-1] *= 0.
        self.B_TM[-1] *= 0.

        # Calculate waves to the right (E_r) and to the left (E_l) using wave function,
        # function of (z, freqs)
        print('Forward and backward waves propagation')
        z_i = 0
        for i, k_z in enumerate(tqdm(
            [k_s_z, k_0_z, k_s_z, k_0_z])):

            z_ip1 = z_i + len(z_local[i])
            z_l_k_z_delay = np.exp(-1j*np.multiply.outer(z_local[i], k_z))
            z_l_k_z_delay_1 = 1/z_l_k_z_delay
            self.E_r_TE[z_i: z_ip1] = z_l_k_z_delay * self.A_TE[i]
            self.E_l_TE[z_i: z_ip1] = z_l_k_z_delay_1 * self.B_TE[i]
            self.E_r_TM[z_i: z_ip1] = z_l_k_z_delay * self.A_TM[i]
            self.E_l_TM[z_i: z_ip1] = z_l_k_z_delay_1 * self.B_TM[i]
            z_i = z_ip1

        # Perform IFFT on E_r, E_l
        # function of (z, time)
        print(f'FFT (size = {self.t.size})')
        if self.is_2D:
            x_delay = np.exp(-2j*np.pi * self.k_prime_r[1] * \
                             np.sqrt(self.eps_rel*self.eps_air*self.mu_0) * \
                             np.multiply.outer(self.x[0], self.freqs))
        else:
            x_delay = 1.
        x_delayed_spect = x_delay * self.spect
        for E in tqdm([self.E_r_TE, self.E_l_TE, self.E_r_TM, self.E_l_TM]):
            np.copyto(E, np.fft.ifft(np.fft.ifftshift(
                E * x_delayed_spect, axes=-1), norm='forward'))

    def start(self):
        self.update_arrays()

        # Define what the animation should do at each frame
        def update_fig2(t_i) -> Iterable[plt.Artist]:
            self.current_frame = t_i
            E_r_TE = np.take(self.E_r_TE, t_i, axis=-1).real
            E_l_TE = np.take(self.E_l_TE, t_i, axis=-1).real
            E_r_TM = np.take(self.E_r_TM, t_i, axis=-1).real
            E_l_TM = np.take(self.E_l_TM, t_i, axis=-1).real
            if self.is_2D:
                self.E_TE_mesh.set_array(np.abs(E_r_TE + E_l_TE))
                self.E_TM_mesh.set_array(np.abs(E_r_TM + E_l_TM))
                self.E_45_mesh.set_array(np.sqrt(2)/2 * \
                    np.abs(E_r_TE + E_l_TE + E_r_TM + E_l_TM))
            else:
                self.E_r_TE_line.set_data(self.z, E_r_TE)
                self.E_l_TE_line.set_data(self.z, E_l_TE)
                self.E_sum_TE_line.set_data(self.z, E_r_TE + E_l_TE)
                self.E_r_TM_line.set_data(self.z, E_r_TM)
                self.E_l_TM_line.set_data(self.z, E_l_TM)
                self.E_sum_TM_line.set_data(self.z, E_r_TM + E_l_TM)
                self.E_r_45_line.set_data(self.z, np.sqrt(2)/2 * \
                    (E_r_TE + E_r_TM))
                self.E_l_45_line.set_data(self.z, np.sqrt(2)/2 * \
                    (E_l_TE + E_l_TM))
                self.E_sum_45_line.set_data(self.z, np.sqrt(2)/2 * \
                    (E_r_TE + E_l_TE + E_r_TM + E_l_TM))
            self.ax2_title.set_text(f't = {self.t[t_i]:.2e} s')
            return (*self.ax2.lines, *self.ax2.collections, *self.ax2.patches, self.ax2_title)
        
        self.anim2 = FuncAnimation(self.fig2, func=update_fig2,
                                   frames=self.t_indices,
                                   interval=20, blit=True)
        self.is_animated = True
        
        # Define display function for sliders and radio buttons update 
        def display(event):
            self.anim2._blit_draw((*self.ax2.lines, *self.ax2.collections, *self.ax2.patches))
            self.anim2._draw_next_frame(self.current_frame, True)

        # Define pause/play function
        def pause_play(event):
            if self.pause_play_button.label.get_text() == 'Pause':
                self.pause_play_button.label.set_text('Play')
                self.anim2.event_source.stop()
            else:
                self.pause_play_button.label.set_text('Pause')
                self.anim2.event_source.start()

        # Define loop back function
        def loop_back(event):
            self.anim2.frame_seq = self.anim2.new_frame_seq()
            self.current_frame = 0
            update_fig2(0)
            self.anim2._draw_next_frame(0, True)
        
        # Assign functions to sliders and buttons
        if self.sliders_and_buttons:
            self.update_slab_slider_cid = self.slab2_slider.on_changed(self.update_arrays)
            self.display_slab_slider_cid = self.slab2_slider.on_changed(display)
            self.update_theta_slider_cid = self.theta_slider.on_changed(self.update_arrays)
            self.display_theta_slider_cid = self.theta_slider.on_changed(display)
            self.display_TE_TM_radio_buttons_cid = self.TE_TM_radio_buttons.on_clicked(display)
            if not self.is_2D:
                self.display_left_right_radio_buttons_cid = self.left_right_radio_buttons.on_clicked(display)
            self.pause_play_button_cid = self.pause_play_button.on_clicked(pause_play)
            self.loop_back_button_cid = self.loop_back_button.on_clicked(loop_back)

            # Change button aspects
            self.start_stop_button.label.set_text('Stop')
            self.pause_play_button.color = '0.85'
            self.pause_play_button.hovercolor = '0.95'
            self.loop_back_button.color = '0.85'
            self.loop_back_button.hovercolor = '0.95'
    
    def stop(self):
        if self.sliders_and_buttons:
            # Disconnect functions from sliders and buttons
            self.slab2_slider.disconnect(self.update_slab_slider_cid)
            self.slab2_slider.disconnect(self.display_slab_slider_cid)
            self.theta_slider.disconnect(self.update_theta_slider_cid)
            self.theta_slider.disconnect(self.display_theta_slider_cid)
            self.TE_TM_radio_buttons.disconnect(self.display_TE_TM_radio_buttons_cid)
            if not self.is_2D:
                self.left_right_radio_buttons.disconnect(self.display_left_right_radio_buttons_cid)
            self.pause_play_button.disconnect(self.pause_play_button_cid)
            self.loop_back_button.disconnect(self.loop_back_button_cid)

            # Change button aspects
            self.start_stop_button.label.set_text('Start')
            self.pause_play_button.label.set_text('Pause')
            self.pause_play_button.color = '0.5'
            self.pause_play_button.hovercolor = '0.5'
            self.loop_back_button.color = '0.5'
            self.loop_back_button.hovercolor = '0.5'

        # Pause animation
        self.anim2.pause()
        self.fig2.canvas.draw()

    # Switch function for 1D/2D figures
    def switch_1D_2D(self):
        plt.close('all')
        self.is_2D = not self.is_2D
        self._init_arrays()
        self._init_figs()
        if self.sliders_and_buttons:
            self._init_sliders_buttons()
    
    # Switch function to update visibility of left-right fields
    def switch_left_right_display(self, label):
        assert self.is_2D == False
        if label == r'$E_+$, $E_-$':
            self.E_r_line.set_visible(True)
            self.E_l_line.set_visible(True)
            self.E_sum_line.set_visible(False)
        elif label == r'$E_{tot}$':
            self.E_r_line.set_visible(False)
            self.E_l_line.set_visible(False)
            self.E_sum_line.set_visible(True)
        else:
            raise ValueError(f'{label} is an invalid label name')
    
    def switch_TE_TM_display(self, label):
        if self.is_2D:
            if label == r'$E_{TE}$':
                self.E_TE_mesh.set_visible(True)
                self.E_TM_mesh.set_visible(False)
                self.E_45_mesh.set_visible(False)
                self.cbar.update_normal(self.E_TE_mesh)
            elif label == r'$E_{TM}$':
                self.E_TE_mesh.set_visible(False)
                self.E_TM_mesh.set_visible(True)
                self.E_45_mesh.set_visible(False)
                self.cbar.update_normal(self.E_TM_mesh)
            elif label == r'$E_{45}$':
                self.E_TE_mesh.set_visible(False)
                self.E_TM_mesh.set_visible(False)
                self.E_45_mesh.set_visible(True)
                self.cbar.update_normal(self.E_45_mesh)
            else:
                raise ValueError(f'{label} is an invalid label name')
        else:
            self.E_l_line.set_visible(False)
            self.E_r_line.set_visible(False)
            self.E_sum_line.set_visible(False)
            if label == r'$E_{TE}$':
                self.E_l_line = self.E_l_TE_line
                self.E_r_line = self.E_r_TE_line
                self.E_sum_line = self.E_sum_TE_line
            elif label == r'$E_{TM}$':
                self.E_l_line = self.E_l_TM_line
                self.E_r_line = self.E_r_TM_line
                self.E_sum_line = self.E_sum_TM_line
            elif label == r'$E_{45}$':
                self.E_l_line = self.E_l_45_line
                self.E_r_line = self.E_r_45_line
                self.E_sum_line = self.E_sum_45_line
            else:
                raise ValueError(f'{label} is an invalid label name')

    def save(self, fig, filename='slabs'):
        f = __file__ + f'/../{filename}'
        if fig == 1:
            self.fig1.savefig(f + '.png', dpi=200)
        elif fig == 2:
            if not self.is_animated:
                raise ValueError("Animation is not started, use start method before saving")
            pil = PillowWriter(fps=20)
            self.anim2.save(f + '.gif', writer=pil)


def pptx_images():
    # Single freq, 2D, theta @ 15°
    # __________________________________________________________________________
    sim = Simulation(single_freq=True, sliders_and_buttons=False, is_2D=True, theta=15)
    sim.start()
    # __________________________________________________________________________
    sim.save(1, 'single_freq_spectrum')
    sim.switch_TE_TM_display(r'$E_{45}$')
    sim.save(2, 'single_freq_2D_E_45')
    sim.switch_TE_TM_display(r'$E_{TE}$')
    sim.save(2, 'single_freq_2D_E_TE')
    sim.switch_TE_TM_display(r'$E_{TM}$')
    sim.save(2, 'single_freq_2D_E_TM')
    # __________________________________________________________________________
    # theta @ brewster_angle
    # __________________________________________________________________________
    theta_b = np.arctan(1/np.sqrt(sim.eps_rel))
    sim.set_theta(theta_b)
    sim.update_arrays()
    sim.switch_TE_TM_display(r'$E_{TM}$')
    sim.save(2, 'single_freq_2D_E_TM_brewster')
    sim.switch_TE_TM_display(r'$E_{TE}$')
    sim.save(2, 'single_freq_2D_E_TE_brewster')
    sim.switch_1D_2D()
    sim.start()
    sim.switch_TE_TM_display(r'$E_{TM}$')
    sim.switch_left_right_display(r'$E_+$, $E_-$')
    sim.save(2, 'single_freq_1D_E_TM_brewster')
    sim.switch_1D_2D()
    sim.start()
    # __________________________________________________________________________
    # Evanescent waves
    # __________________________________________________________________________
    theta_c = np.arcsin(1/np.sqrt(sim.eps_rel))
    sim.set_theta(theta_c + 5)
    sim.update_arrays()
    sim.switch_TE_TM_display(r'$E_{45}$')
    sim.save(2, 'single_freq_2D_E_45_evan_far')
    original_pos = sim.slab2_start
    sim.change_slab2_pos(sim.slab1_end + 0.2)
    sim.update_arrays()
    sim.save(2, 'single_freq_2D_E_45_evan_close')
    sim.switch_1D_2D()
    sim.start()
    sim.switch_TE_TM_display(r'$E_{45}$')
    sim.switch_left_right_display(r'$E_{tot}$')
    sim.save(2, 'single_freq_1D_E_45_evan_close')
    sim.change_slab2_pos(original_pos)
    sim.update_arrays()
    sim.save(2, 'single_freq_1D_E_45_evan_far')
    # __________________________________________________________________________
    # Wavepacket
    # __________________________________________________________________________
    plt.close('all')
    sim2 = Simulation(single_freq=False, sliders_and_buttons=False, is_2D=True, theta=15)
    sim2.start()
    sim2.save(1, 'wave_packet_spectrum')
    sim2.switch_TE_TM_display(r'$E_{45}$')
    sim2.save(2, 'wave_packet_2D_E_45')
    sim2.switch_TE_TM_display(r'$E_{TE}$')
    sim2.save(2, 'wave_packet_2D_E_TE')
    sim2.switch_TE_TM_display(r'$E_{TM}$')
    sim2.save(2, 'wave_packet_2D_E_TM')


def parse_args(arg_list: list[str] = None):
    parser = argparse.ArgumentParser(
        description='Run a simulation of EM waves through slabs with different permittivities',
        usage='python %(prog)s [SIMULATION_PARAMETERS]')
    
    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument('-s', '--single_freq', action='store_true',
                           help='Simulation with a single frequency')
    sim_group.add_argument('-wp', '--wave_packet', action='store_true',
                           help='Simulation with multiple frequencies')
    sim_group.add_argument('-f_0', action='store', type=float, default=None,
                           help='Sets the center frequency to this value')
    
    myParser = parser.parse_args(arg_list)
    myParser.single_freq = not myParser.wave_packet

    return myParser


def main(arg_list: list[str] = None):
    args = parse_args(arg_list)
    # pptx_images()
    sim = Simulation(**vars(args), sliders_and_buttons=True, is_2D=True)
    plt.show()


if __name__ == '__main__':
    main()

