# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:36:18 2025

@author: carbonnelleg
"""
from typing import Iterable
import argparse
import numpy as np
from scipy.constants import epsilon_0, mu_0
from scipy import signal as sg
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons

"""
_______________________________________________________________________________
Declaring variables
"""
z_min = 0.
z_max = 10.
z_step = 4.0e-3

slab1_start, slab1_end = 2., 4.
slab2_start, slab2_end = 6., 8.

t_min = 0.
t_max = 1.6e-7
t_step = 1.0e-10

f_0 = 3.0e9
sigma_f_0 = 0.3e9

eps_rel = 4.
"""
End of variable declaration
_______________________________________________________________________________
"""


class Simulation:

    def __init__(self,
                 # Space grid initialization
                 z_min=z_min,
                 z_max=z_max,
                 z_step=z_step,
                 # Initial positions of the slabs
                 slab1_start=slab1_start,
                 slab1_end=slab1_end,
                 slab2_start=slab2_start,
                 slab2_end=slab2_end,
                 # Time grid initialization
                 t_min=t_min,
                 t_max=t_max,
                 t_step=t_step,
                 # Spectrum initialization
                 single_freq=False,
                 f_0=f_0,
                 sigma_f_0=sigma_f_0,
                 # Physical parameters
                 eps_air=epsilon_0,
                 eps_rel=eps_rel,
                 mu_0=mu_0,
                 **kwargs):
        # Space grid initialization
        self.z_min = z_min
        self.z_max = z_max
        self.z_step = z_step
        self.z = np.arange(z_min, z_max, z_step)
        # Initial positions of the slabs
        self.slab1_start = slab1_start
        self.slab1_end = slab1_end
        self.slab2_start = slab2_start
        self.slab2_end = slab2_end
        # Time grid initialization
        self.t_min = t_min
        self.t_max = t_max
        self.t_step = t_step
        self.t = np.arange(t_min, t_max, t_step)
        self.t_indices = np.arange(self.t.size)
        # Spectrum initialization
        self.single_freq = single_freq
        self.f_s = 1/t_step
        self.f_step = self.f_s/self.t.size
        self.freqs = np.fft.ifftshift(
            np.fft.fftfreq(self.t.size, d=self.t_step))
        if f_0 is None:
            if single_freq: f_0 = .5e9
            else: f_0 = 3.e9
        self.f_0 = f_0
        if single_freq:
            # FT of single frequency is 2 deltas at +/- f_0
            self.spect = 1/2 * \
                (np.abs(np.abs(self.freqs) - f_0) < self.f_step/2)
        else:
            # Gaussian window, shifted around +/- f_0, normalized
            self.spect = sg.windows.gaussian(
                self.t.size, std=sigma_f_0/self.f_step)
            self.spect = np.convolve(
                np.abs(np.abs(self.freqs) - f_0) < self.f_step/2,
                self.spect, mode='same')
            self.spect *= self.f_step/(2*sigma_f_0*np.sqrt(2*np.pi))
        # Physical parameters
        self.eps_air = eps_air
        self.eps_rel = eps_rel
        self.eps_slab = eps_rel * eps_air
        self.mu_0 = mu_0
        # Impedances
        self.eta = np.sqrt(mu_0/eps_air)
        self.eta_prime = self.eta / np.sqrt(eps_rel)
        # Reflexion and transmission coefficients
        self.rho = (self.eta_prime - self.eta) / (self.eta_prime + self.eta)
        self.tau = 1 + self.rho
        self.rho_prime = -self.rho
        self.tau_prime = 1 + self.rho_prime
        # Initialize arrays
        self._init_arrays(**kwargs)
        # Initialize matplotlib figures
        self._init_figs(**kwargs)

    def _init_arrays(self, **kwargs):
        # Arrays function of (z)
        self.is_slab = ((self.z > self.slab1_start) & (self.z < self.slab1_end)) | \
                       ((self.z > self.slab2_start) & (self.z < self.slab2_end))
        # Arrays function of (freqs)
        self.k_air = 2*np.pi*self.freqs*np.sqrt(self.eps_air * self.mu_0)
        self.k_slab = 2*np.pi*self.freqs*np.sqrt(self.eps_slab * self.mu_0)
        self.Gamma = np.zeros_like(self.freqs, dtype=np.complex128)
        self.A = np.zeros((5, self.freqs.size), dtype=np.complex128)
        self.A[0] += 1.
        self.B = np.zeros((5, self.freqs.size), dtype=np.complex128)
        # Arrays function of (z, freqs)
        self.E_r = np.zeros((self.z.size, self.freqs.size),
                            dtype=np.complex128)
        self.E_l = np.zeros((self.z.size, self.freqs.size),
                            dtype=np.complex128)

    def _init_figs(self, **kwargs):
        # Figure of the spectrum
        fig, ax = plt.subplots(num='Spectrum')
        ax.stem(2*np.pi*self.freqs, self.spect,
                label=r'$\hat{E}_{0+} (z=0, \omega)$',
                markerfmt='b.', linefmt='blue')
        ax.set_xlabel(r'$\omega$ [rad/s]')
        ax.set_xticks(2*np.pi*np.array([0, self.f_0, -self.f_0]),
                      labels=[0, r'$\omega_0$', r'$-\omega_0$'])
        ax.legend()
        ax.set_title(fr'$f_0$ = {self.f_0:.1e}')
        fig.suptitle('Frequency domain')
        self.fig1 = fig
        self.ax1 = ax

        # Animated figure of the waves through slabs
        fig, ax = plt.subplots(num='Waves through slabs')
        fig.subplots_adjust(right=0.8, bottom=0.2)
        # Initialize all artists
        self.slab_poly = ax.fill_between(
            self.z, -2., 2., where=self.is_slab, facecolor='grey',
            alpha=.5, label=r'Slab ($\varepsilon_r$ = 'f'{self.eps_rel:.2f})')
        self.E_r_line, = ax.plot([], [], color='blue', label=r'$E_+$')
        self.E_l_line, = ax.plot([], [], color='red', label=r'$E_-$')
        self.E_sum_line, = ax.plot([], [], color='green', visible=False,
                                   label=r'$E = E_+ + E_-$')
        self.E_sum_line.remove()
        # Initialize slider
        ax_slider = fig.add_axes([0.2, 0.05, 0.65, 0.03])
        self.slider = Slider(
            ax_slider, label='Slab 2 Start Pos',
            valmin=self.slab1_end + self.z_step,
            valmax=self.z_max - self.slab2_end + self.slab2_start,
            valinit=self.slab2_start)
        # Initialize restart button
        ax_restart_button = fig.add_axes([0.81, 0.2, 0.1, 0.05])
        self.restart_button = Button(ax_restart_button, 'Restart')
        # Initialize toggle button (Switch between E_r & E_l vs E_sum)
        ax_check_buttons = fig.add_axes([0.81, 0.4, 0.18, 0.15])
        self.check_buttons = CheckButtons(
            ax_check_buttons, actives=[True, True, False],
            labels=[r'$E_+$', r'$E_-$', r'$E = E_+ + E_-$'])
        # Other figure features
        ax.set_xlabel('Distance [m]')
        ax.set_xlim([self.z_min, self.z_max])
        if self.single_freq:
            ax.set_ylim([-2.2, 2.2])
        else:
            ax.set_ylim([-1.1, 1.1])
        self.fig2 = fig
        self.ax2 = ax

    def update_slab(self):
        """
        Updates the different fields and waves of the simulation for the current
        slab positions.

        Returns
        -------
        is_slab : np.ndarray[bool], shape = (z.size,)
            True where z is inside a slab, function of (z)
        E_r : np.ndarray[np.complex128], shape = (z.size, t.size)
            Electric field of the wave going to the right, function of (z, t)
        E_l : np.ndarray[np.complex128], shape = (z.size, t.size)
            Electric field of the wave going to the left, function of (z, t)
        """
        (d1, d2, d3, d4) = (self.slab1_start,
                            self.slab1_end, self.slab2_start, self.slab2_end)
        z = self.z
        (k_0, k_s) = (self.k_air, self.k_slab)
        (r_0, r_s) = (self.rho, self.rho_prime)
        (t_0, t_s) = (self.tau, self.tau_prime)

        # Update slab region, function of (z)
        self.is_slab = ((z > d1) & (z < d2)) | ((z > d3) & (z < d4))

        # Define local space variables for 5 consecutive layers
        z_local = [
            z[(z <= d1).nonzero()],
            z[((z > d1) & (z < d2)).nonzero()] - d1,
            z[((z >= d2) & (z <= d3)).nonzero()] - d2,
            z[((z > d3) & (z < d4)).nonzero()] - d3,
            z[(z >= d4).nonzero()] - d4
        ]

        # Calculate Gamma at z=0, function of (freqs)
        self.Gamma *= 0
        for (k, l, r) in zip(
                [k_s, k_0, k_s, k_0],
                [d4 - d3, d3 - d2, d2 - d1, d1],
                [r_s, r_0, r_s, r_0]):

            self.Gamma = np.exp(-2j*k*l) * \
                (r + self.Gamma) / (1 + r*self.Gamma)

        # Calculate 5 forward (A) and backward (B) fields at the start of each z_local,
        # function of (freqs)
        self.B[0] = self.A[0] * self.Gamma
        for i, (k, l, r, t) in enumerate(zip(
                [k_0, k_s, k_0, k_s],
                [d1, d2 - d1, d3 - d2, d4 - d3],
                [r_s, r_0, r_s, r_0],
                [t_s, t_0, t_s, t_0])):

            self.A[i+1] = 1/t * np.exp(-1j*k*l) * \
                self.A[i] + r/t * np.exp(1j*k*l) * self.B[i]
            self.B[i+1] = r/t * np.exp(-1j*k*l) * \
                self.A[i] + 1/t * np.exp(1j*k*l) * self.B[i]

        # Calculate waves to the right (E_r) and to the left (E_l) using wave function,
        # function of (freqs, z_local)
        z_i = 0
        for i, k in enumerate([k_0, k_s, k_0, k_s, k_0]):

            self.E_r[z_i: z_i+len(z_local[i])] = self.A[i] * \
                np.exp(-1j*np.outer(k, z_local[i])).T
            self.E_l[z_i: z_i+len(z_local[i])] = self.B[i] * \
                np.exp(1j*np.outer(k, z_local[i])).T
            z_i += len(z_local[i])

        # Concatenate consecutive waves and perform IFFT
        self.E_r = np.fft.ifft(np.fft.ifftshift(
            self.E_r * self.spect, axes=1), norm='forward')
        self.E_l = np.fft.ifft(np.fft.ifftshift(
            self.E_l * self.spect, axes=1), norm='forward')
        
        return self.is_slab, self.E_r, self.E_l

    def start(self):
        self.update_slab()   
        # Define what the slider update should do
        def update_slider(val):
            self.slab2_end = val + self.slab2_end - self.slab2_start
            self.slab2_start = val
            self.update_slab()
            self.slab_poly.set_data(self.z, -2., 2., where=self.is_slab)
            self.fig2.canvas.draw_idle()

        # Define what the animation should do at each frame
        def update_fig(t_i) -> Iterable[plt.Artist]:
            E_r = self.E_r[:, t_i].real
            E_l = self.E_l[:, t_i].real
            self.E_r_line.set_data(self.z, E_r)
            self.E_l_line.set_data(self.z, E_l)
            self.E_sum_line.set_data(self.z, E_r + E_l)
            self.ax2.legend(loc=1)
            self.fig2.suptitle(f't = {self.t[t_i]:.2e} s')
            return self.E_r_line, self.E_l_line, self.E_sum_line

        # Create the animation
        anim = FuncAnimation(self.fig2, update_fig, frames=self.t_indices,
                             interval=10)
        
        # Define restart function
        def restart(event):
            anim.event_source.stop()
            anim.frame_seq = anim.new_frame_seq()
            anim.event_source.start()
        
        # Define toggle function to update visibility
        def toggle_display(label):
            if label == r'$E_+$':
                if self.E_r_line.get_visible():
                    self.E_r_line.remove()
                else:
                    self.ax2.add_artist(self.E_r_line)
                self.E_r_line.set_visible(not self.E_r_line.get_visible())
            elif label == r'$E_-$':
                if self.E_l_line.get_visible():
                    self.E_l_line.remove()
                else:
                    self.ax2.add_artist(self.E_l_line)
                self.E_l_line.set_visible(not self.E_l_line.get_visible())
            elif label == r'$E = E_+ + E_-$':
                if self.E_sum_line.get_visible():
                    self.E_sum_line.remove()
                else:
                    self.ax2.add_artist(self.E_sum_line)
                self.E_sum_line.set_visible(not self.E_sum_line.get_visible())
            self.fig2.canvas.draw_idle()
                
        # Assign functions to slider and buttons
        self.slider.on_changed(update_slider)
        self.restart_button.on_clicked(restart)
        self.check_buttons.on_clicked(toggle_display)

        plt.show()


def parse_args(arg_list: list[str] = None):
    parser = argparse.ArgumentParser(
        description='Run a simulation of EM waves through slabs with different permittivities',
        usage='python slabs.py [SIMULATION_PARAMETERS]')
    
    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument('-0', '--single_freq', action='store_true',
                           help='Simulation with a single frequency')
    sim_group.add_argument('-1', '--wave_packet', action='store_true',
                           help='Simulation with multiple frequencies')
    sim_group.add_argument('-f_0', action='store', type=float, default=None,
                           help='Sets the center frequency to this value')
    
    return parser.parse_args(arg_list)


def main(arg_list: list[str] = None):
    args = parse_args(arg_list)
    sim = Simulation(**vars(args))
    sim.start()


if __name__ == '__main__':
    main()
