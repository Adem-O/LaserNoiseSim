#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ipywidgets import interact, FloatSlider, Checkbox, IntText, FloatText, Layout, HBox, VBox

# Define the power law PSD contributions and Gaussian peak
def generate_power_law_psd(frequencies, exponent, amplitude=1.0, randomness=0.1):
    psd = np.zeros_like(frequencies)
    non_zero_frequencies = frequencies > 0
    base_psd = amplitude * (frequencies[non_zero_frequencies]**exponent)
    random_factor = 1 + randomness * (np.random.randn(len(base_psd)))
    psd[non_zero_frequencies] = base_psd * random_factor
    return psd

def add_gaussian_peak(frequencies, peak_location, peak_amplitude, peak_randomness, peak_width):
    psd_peak = np.zeros_like(frequencies)
    gaussian_shape = peak_amplitude * np.exp(-((frequencies - peak_location)**2) / (2 * peak_width**2))
    random_factor = 1 + peak_randomness * np.random.randn(len(frequencies))
    psd_peak = gaussian_shape * random_factor
    return psd_peak

# Corrected autocorrelation PSD function with proper normalization
def compute_autocorrelation_psd(electric_field, dt):
    N = len(electric_field)
    R_E = np.correlate(electric_field, electric_field, mode='full')[N-1:]
    autocorr_laser_psd = np.abs(rfft(R_E)) * dt / N
    return autocorr_laser_psd

# Compute the PSD and autocorrelation-based PSD
def compute_psd(N, T, f_central, include_f3, amp_f3, rand_f3,
                include_f2, amp_f2, rand_f2, 
                include_f1, amp_f1, rand_f1, 
                include_f0, amp_f0, rand_f0,
                include_fp1, amp_fp1, rand_fp1,
                include_fp2, amp_fp2, rand_fp2,
                include_peak, peak_location, peak_amplitude, peak_randomness, peak_width):
    
    dt = T / N
    frequencies = rfftfreq(N, dt)

    # Ensure f_central and peak_location are within allowable range
    f_min = 1 / T
    f_max = N / (2 * T)
    f_central = min(max(f_central, f_min), f_max)
    peak_location = min(max(peak_location, f_min), f_max)
    
    psd_contributions = []

    # Add selected PSD contributions
    if include_f3:
        psd_contributions.append(generate_power_law_psd(frequencies, exponent=-3, amplitude=amp_f3, randomness=rand_f3))
    if include_f2:
        psd_contributions.append(generate_power_law_psd(frequencies, exponent=-2, amplitude=amp_f2, randomness=rand_f2))
    if include_f1:
        psd_contributions.append(generate_power_law_psd(frequencies, exponent=-1, amplitude=amp_f1, randomness=rand_f1))
    if include_f0:
        psd_contributions.append(generate_power_law_psd(frequencies, exponent=0, amplitude=amp_f0, randomness=rand_f0))
    if include_fp1:
        psd_contributions.append(generate_power_law_psd(frequencies, exponent=1, amplitude=amp_fp1, randomness=rand_fp1))
    if include_fp2:
        psd_contributions.append(generate_power_law_psd(frequencies, exponent=2, amplitude=amp_fp2, randomness=rand_fp2))

    if psd_contributions:
        total_psd = np.sum(psd_contributions, axis=0)
    else:
        total_psd = np.ones_like(frequencies) * 1e-12

    if include_peak:
        peak_psd = add_gaussian_peak(frequencies, peak_location, peak_amplitude, peak_randomness, peak_width)
        total_psd += peak_psd
    
    noise_freq_domain = (np.random.normal(size=len(frequencies)) + 1j * np.random.normal(size=len(frequencies))) * np.sqrt(total_psd)
    noise_time_domain = irfft(noise_freq_domain, N)
    time = np.arange(N) * dt
    electric_field = np.cos(2 * np.pi * f_central * time + noise_time_domain)
    
    electric_field_fft = np.abs(rfft(electric_field))**2
    laser_psd = electric_field_fft / N
    autocorr_laser_psd = compute_autocorrelation_psd(electric_field, dt)

    plot_all_subplots(frequencies, total_psd, time, electric_field, laser_psd, autocorr_laser_psd, f_central)

# Plotting functions with GridSpec for layout flexibility
def plot_all_subplots(frequencies, total_psd, time, electric_field, laser_psd, autocorr_laser_psd, f_central):
    fig = plt.figure(figsize=(14, 12),layout='tight')
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1])

    # PSD plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.loglog(frequencies[frequencies > 0], total_psd[frequencies > 0])
    ax1.set_title('Simulated Frequency Noise PSD')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD')
    ax1.grid(True)

    # Electric field plot
    ax2 = fig.add_subplot(gs[0, 1])
    num_periods = 30
    max_time = num_periods / f_central
    time_points = time[time <= max_time]
    electric_field_points = electric_field[:len(time_points)]
    if len(electric_field_points) > 1:
        ax2.plot(time_points, electric_field_points)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for 30 periods\nTry adjusting parameters', ha='center', va='center')
    ax2.set_title('Time-Dependent Electric Field (Noisy Laser)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Electric Field Amplitude')
    ax2.grid(True)

    # Laser PSD plot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.loglog(frequencies[frequencies > 0], laser_psd[frequencies > 0])
    ax3.set_title('PSD of the Noisy Laser')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power Spectral Density')
    ax3.grid(True)

    # Autocorrelation-based PSD plot
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.loglog(frequencies[frequencies > 0], autocorr_laser_psd[frequencies > 0])
    ax4.set_title('PSD from Autocorrelation Function')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power Spectral Density')
    ax4.grid(True)

    # Combined plot of Autocorrelation PSD and Laser PSD (Zoomed around central frequency ± 0.5 MHz)
    ax5 = fig.add_subplot(gs[2, :])
    idx = (np.abs(frequencies - f_central)).argmin()
    idr = len(frequencies)*0.005
    idl = idx-int(idr/2)
    idu = idx+int(idr/2)
    ax5.plot(frequencies[idl:idu], laser_psd[idl:idu], label="Laser PSD")
    ax5.plot(frequencies[idl:idu], autocorr_laser_psd[idl:idu], label="Autocorr PSD", linestyle='--')
    ax5.set_yscale('log')
    ax5.set_title('Comparison of Autocorrelation PSD and Laser PSD (Zoomed)')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Power Spectral Density')
    ax5.legend()
    ax5.grid(True)

    plt.show()

# Widget layout and interaction setup
def update_widgets(N, T):
    f_min = 1 / T
    f_max = N / (2 * T)
    f_central_slider = FloatSlider(min=f_min, max=f_max, step=f_min, value=f_min, description="Central Freq (Hz):", layout=Layout(width='300px'))
    peak_location_slider = FloatSlider(min=f_min, max=f_max, step=f_min, value=f_min, description="Peak Location (Hz):", layout=Layout(width='300px'))
    return f_central_slider, peak_location_slider

N_widget = IntText(value=4096, description="Num Time Points:", layout=Layout(width='300px'))
T_widget = FloatText(value=1e-3, description="Total Time (s):", layout=Layout(width='300px'))

# Display widgets with interactive controls
def display_widgets():
    f_central_slider, peak_location_slider = update_widgets(N_widget.value, T_widget.value)
    interact(
        compute_psd,
        N=N_widget,
        T=T_widget,
        f_central=f_central_slider,
        include_f3=Checkbox(value=False, description="Include $f^{-3}$"),
        amp_f3=FloatText(min=1e-4, max=10, step=1e-4, value=1e-2, description="Amp $f^{-3}$", layout=Layout(width='300px')),
        rand_f3=FloatSlider(min=0, max=0.2, step=0.01, value=0.1, description="Rand $f^{-3}$", layout=Layout(width='300px')),
        include_f2=Checkbox(value=False, description="Include $f^{-2}$"),
        amp_f2=FloatText(min=1e-4, max=10, step=1e-4, value=1e-2, description="Amp $f^{-2}$", layout=Layout(width='300px')),
        rand_f2=FloatSlider(min=0, max=0.2, step=0.01, value=0.1, description="Rand $f^{-2}$", layout=Layout(width='300px')),
        include_f1=Checkbox(value=False, description="Include $f^{-1}$"),
        amp_f1=FloatText(min=1e-4, max=10, step=1e-4, value=1e-3, description="Amp $f^{-1}$", layout=Layout(width='300px')),
        rand_f1=FloatSlider(min=0, max=0.2, step=0.01, value=0.1, description="Rand $f^{-1}$", layout=Layout(width='300px')),
        include_f0=Checkbox(value=False, description="Include $f^0$"),
        amp_f0=FloatText(min=1e-8, max=10, step=1e-4, value=1e-4, description="Amp $f^0$", layout=Layout(width='300px')),
        rand_f0=FloatSlider(min=0, max=0.2, step=0.01, value=0.1, description="Rand $f^0$", layout=Layout(width='300px')),
        include_fp1=Checkbox(value=False, description="Include $f^{1}$"),
        amp_fp1=FloatText(min=1e-8, max=10, step=1e-4, value=1e-4, description="Amp $f^{1}$", layout=Layout(width='300px')),
        rand_fp1=FloatSlider(min=0, max=0.2, step=0.01, value=0.1, description="Rand $f^{1}$", layout=Layout(width='300px')),
        include_fp2=Checkbox(value=False, description="Include $f^{2}$"),
        amp_fp2=FloatText(min=1e-8, max=10, step=1e-4, value=1e-4, description="Amp $f^{2}$", layout=Layout(width='300px')),
        rand_fp2=FloatSlider(min=0, max=0.2, step=0.01, value=0.1, description="Rand $f^{2}$", layout=Layout(width='300px')),
        include_peak=Checkbox(value=False, description="Include Noise Peak"),
        peak_location=peak_location_slider,
        peak_amplitude=FloatSlider(min=1e-8, max=1e4, step=1e-4, value=1, description="Peak Amplitude", layout=Layout(width='300px')),
        peak_randomness=FloatSlider(min=0, max=1, step=0.01, value=0.1, description="Peak Randomness", layout=Layout(width='300px')),
        peak_width=FloatSlider(min=1e3, max=1e6, step=1e3, value=1e4, description="Peak Width (Hz)", layout=Layout(width='300px'))
    )

# Observers to update widget values
N_widget.observe(lambda change: display_widgets(), names='value')
T_widget.observe(lambda change: display_widgets(), names='value')

# Initial display
display_widgets()

