#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.fft import irfft, rfftfreq, fft, fftshift, fftfreq
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ipywidgets import (
    FloatSlider, Checkbox, IntText, FloatText, Layout, HBox, VBox, Output
)
from IPython.display import display, clear_output
from matplotlib.patches import ConnectionPatch  # Needed for connection lines

# Global variables to store computed data
computed_data = {}

# Define the power law PSD contributions and Gaussian peak
def generate_power_law_psd(frequencies, exponent, amplitude=1.0):
    """
    Generates a power-law PSD with fixed randomness.

    Parameters:
    - frequencies: Array of frequency values.
    - exponent: Exponent of the power-law.
    - amplitude: Amplitude of the PSD.

    Returns:
    - psd: Power Spectral Density array.
    """
    psd = np.zeros_like(frequencies)
    non_zero_frequencies = frequencies > 0
    base_psd = amplitude * (frequencies[non_zero_frequencies] ** exponent)
    randomness = 0.2
    # Generate noise in frequency domain
    random_factor = 1 + randomness * np.random.randn(len(frequencies[non_zero_frequencies]))
    psd[non_zero_frequencies] = base_psd * random_factor + 1e-18
    return psd

def add_gaussian_peak(frequencies, peak_location, peak_amplitude, peak_width):
    """
    Adds a Gaussian peak to the PSD with fixed randomness.

    Parameters:
    - frequencies: Array of frequency values.
    - peak_location: Central frequency of the peak.
    - peak_amplitude: Amplitude of the peak.
    - peak_width: Width of the peak.

    Returns:
    - psd_peak: Gaussian peak PSD array.
    """
    # Fixed peak randomness set to 0.2
    peak_randomness = 0.2
    gaussian_shape = peak_amplitude * np.exp(-((frequencies - peak_location) ** 2) / (2 * peak_width ** 2))
    random_factor = 1 + peak_randomness * np.random.randn(len(frequencies))
    psd_peak = gaussian_shape * random_factor + 1e-18 * random_factor
    return psd_peak

# Compute the autocorrelation function based on the given integral formula
def compute_autocorrelation_function(frequencies, total_psd, tau_array, f_central):
    """
    Computes the autocorrelation function R_E(tau) based on the integral formula.

    Parameters:
    - frequencies: Array of frequency values.
    - total_psd: Total frequency noise PSD (S_{delta nu}(f)).
    - tau_array: Array of tau values.
    - f_central: Central frequency of the laser.

    Returns:
    - R_E_tau: Autocorrelation function array.
    """
    delta_f = frequencies[1] - frequencies[0]  # Frequency spacing
    phi_tau = np.zeros_like(tau_array, dtype=np.float64)

    # Compute phi(tau) for each tau
    for idx, tau in enumerate(tau_array):
        # Compute sin^2(pi * f * tau) / f^2
        sin_term = np.sin(np.pi * frequencies * tau)
        sin_squared = sin_term ** 2
        f_squared = frequencies ** 2
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            integrand = np.where(frequencies > 0, total_psd * sin_squared / f_squared, 0.0)
        # Numerically integrate over frequency
        phi_tau[idx] = 2 * np.sum(integrand) * delta_f

    # Compute R_E(tau) using f_central as nu_0
    R_E_tau = np.exp(1j * 2 * np.pi * f_central * tau_array) * np.exp(-phi_tau)

    return R_E_tau

# Compute the PSD and autocorrelation-based PSD
def compute_psd(
    N, T, f_central, include_f3, amp_f3,
    include_f2, amp_f2, 
    include_f1, amp_f1, 
    include_f0, amp_f0,
    include_fp1, amp_fp1,
    include_fp2, amp_fp2,
    include_peak, peak_location, peak_amplitude, peak_width
):
    """
    Computes the PSD based on user-selected parameters.

    Parameters:
    - N: Number of time points.
    - T: Total time duration.
    - f_central: Central frequency of the electric field.
    - include_f3 to include_fp2: Booleans to include specific PSD contributions.
    - amp_f3 to amp_fp2: Amplitudes for specific PSD contributions.
    - include_peak: Boolean to include a Gaussian noise peak.
    - peak_location: Central frequency of the noise peak.
    - peak_amplitude: Amplitude of the noise peak.
    - peak_width: Width of the noise peak.

    Returns:
    - data_dict: Dictionary containing computed data.
    """
    dt = T / N
    frequencies = rfftfreq(N, dt)

    # Ensure f_central and peak_location are within allowable range
    f_min = 1 / T
    f_max = N / (2 * T)
    f_central = np.clip(f_central, f_min, f_max)
    peak_location = np.clip(peak_location, f_min, f_max)
    
    psd_contributions = []

    # Add selected PSD contributions with labels
    if include_f3:
        psd_contributions.append( ('f⁻³', generate_power_law_psd(frequencies, exponent=-3, amplitude=amp_f3)) )
    if include_f2:
        psd_contributions.append( ('f⁻²', generate_power_law_psd(frequencies, exponent=-2, amplitude=amp_f2)) )
    if include_f1:
        psd_contributions.append( ('f⁻¹', generate_power_law_psd(frequencies, exponent=-1, amplitude=amp_f1)) )
    if include_f0:
        psd_contributions.append( ('f⁰', generate_power_law_psd(frequencies, exponent=0, amplitude=amp_f0)) )
    if include_fp1:
        psd_contributions.append( ('f¹', generate_power_law_psd(frequencies, exponent=1, amplitude=amp_fp1)) )
    if include_fp2:
        psd_contributions.append( ('f²', generate_power_law_psd(frequencies, exponent=2, amplitude=amp_fp2)) )

    if include_peak:
        psd_contributions.append( ('Noise Peak', add_gaussian_peak(frequencies, peak_location, peak_amplitude, peak_width)) )

    if psd_contributions:
        total_psd = np.sum([psd for label, psd in psd_contributions], axis=0)
    else:
        total_psd = np.ones_like(frequencies) * 1e-12  # Minimal PSD if no contributions are selected

    # Generate frequency noise in frequency domain
    # Random phase with the specified PSD
    phase_noise_freq_domain = np.sqrt(total_psd) * (np.random.randn(len(frequencies)) + 1j * np.random.randn(len(frequencies)))
    # Convert to time domain
    phase_noise_time_domain = irfft(phase_noise_freq_domain, n=N)

    # Generate electric field with noise
    time = np.arange(N) * dt
    electric_field = np.cos(2 * np.pi * f_central * time + phase_noise_time_domain)

    # Compute autocorrelation-based PSD using the autocorrelation function
    tau_array = np.arange(-N//2, N//2) * dt
    R_E_tau = compute_autocorrelation_function(frequencies, total_psd, tau_array, f_central)
    autocorr_laser_psd_full = np.abs(fftshift(fft(R_E_tau)))**2
    freqs_fft = fftshift(fftfreq(len(R_E_tau), dt))
    positive_freqs = freqs_fft > 0
    autocorr_laser_psd = autocorr_laser_psd_full[positive_freqs]
    freqs_plot = freqs_fft[positive_freqs]

    # Store computed data in a dictionary
    data_dict = {
        'frequencies': frequencies,
        'total_psd': total_psd,
        'time': time,
        'electric_field': electric_field,
        'autocorr_laser_psd': autocorr_laser_psd,
        'freqs_autocorr_psd': freqs_plot,
        'f_central': f_central,
        'psd_contributions': psd_contributions
    }

    return data_dict

def plot_all_subplots(
    data_dict, show_all, show_beta_line
):
    frequencies = data_dict['frequencies']
    total_psd = data_dict['total_psd']
    time = data_dict['time']
    electric_field = data_dict['electric_field']
    autocorr_laser_psd = data_dict['autocorr_laser_psd']
    freqs_autocorr_psd = data_dict['freqs_autocorr_psd']
    f_central = data_dict['f_central']
    psd_contributions = data_dict['psd_contributions']
    
    # Calculate instantaneous linewidth if white noise (f^0) is included
    linewidth_instantaneous = None
    for label, psd in psd_contributions:
        if label == 'f⁰':
            S_0 = psd[frequencies > 0][0]  # White noise amplitude
            linewidth_instantaneous = 1/2* np.pi * S_0
            break

    fig = plt.figure(figsize=(14, 14), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1.5])

    # Simulated Frequency Noise PSD plot spanning two columns
    ax1 = fig.add_subplot(gs[0, :])

    if show_all and psd_contributions:
        for label, psd in psd_contributions:
            ax1.loglog(frequencies[frequencies > 0], psd[frequencies > 0], alpha=0.6, label=label)  
        ax1.legend(fontsize=14)

    total_alpha = 0.6 if show_all else 1.0
    ax1.loglog(frequencies[frequencies > 0], total_psd[frequencies > 0], label='Total PSD', linewidth=2, color='tab:blue', alpha=total_alpha)

    if show_beta_line:
        beta_line_psd = np.zeros_like(frequencies)
        positive_freqs = frequencies > 0
        beta_line_psd[positive_freqs] = (8 * np.log(2) * frequencies[positive_freqs]) / (np.pi**2)
        ax1.loglog(frequencies[positive_freqs], beta_line_psd[positive_freqs], linestyle='--', color='green', label='Beta-separation line')
        ax1.legend(fontsize=14)

        idx_beta = np.where(total_psd[positive_freqs] <= beta_line_psd[positive_freqs])[0]
        if len(idx_beta) > 0:
            f_beta = frequencies[positive_freqs][idx_beta[0]]
            ax1.axvline(f_beta, color='red', linestyle='--', label=f'$f_\\beta$ = {f_beta:.2e} Hz')
            ax1.legend(fontsize=14)
            idxs = np.where((frequencies >= 0) & (frequencies <= f_beta))[0]
            linewidth_beta = np.sqrt(8*np.log(2)*cumulative_trapezoid(total_psd[idxs], frequencies[idxs], initial=0)[-1])
            indices = np.where((frequencies <= f_beta) & (frequencies > 0))
            ymin_fill = ax1.get_ylim()[0]
            ax1.fill_between(frequencies[indices], total_psd[indices], ymin_fill, color='orange', alpha=0.3)
        else:
            f_beta = None
            linewidth_beta = None
    else:
        f_beta = None
        linewidth_beta = None

    # Update title with estimated linewidths if applicable
    title_text = 'Simulated Frequency Noise PSD'
    if show_beta_line and linewidth_beta is not None:
        title_text += f'\nEstimated Linewidth (beta-line): {linewidth_beta:.2e} Hz'
    if linewidth_instantaneous is not None:
        title_text += f'\nInstantaneous Linewidth: {linewidth_instantaneous:.2e} Hz'
    ax1.set_title(title_text, fontsize=18)
    ax1.set_xlabel('Frequency (Hz)', fontsize=15)
    ax1.set_ylabel(r'$S_{\delta\nu}(f)~(\mathrm{Hz}^2/\mathrm{Hz})$', fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(True)

    # PSD from Autocorrelation Function plot (now in place of PSD of the noisy laser)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.loglog(freqs_autocorr_psd[freqs_autocorr_psd > 0], autocorr_laser_psd[freqs_autocorr_psd > 0])
    ax3.set_title('PSD from Autocorrelation Function', fontsize=18)
    ax3.set_xlabel('Frequency (Hz)', fontsize=15)
    ax3.set_ylabel('Power Spectral Density', fontsize=15)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.grid(True)

    # Time-Dependent Electric Field plot (moved down)
    ax4 = fig.add_subplot(gs[1, 1])
    num_periods = 30
    max_time = num_periods / f_central
    time_points = time[time <= max_time]
    electric_field_points = electric_field[:len(time_points)]
    if len(electric_field_points) > 1:
        ax4.plot(time_points, electric_field_points)
    else:
        ax4.text(
            0.5, 0.5, 
            'Insufficient data for 30 periods\nTry adjusting parameters', 
            ha='center', va='center'
        )
    ax4.set_title('Time-Dependent Electric Field (Noisy Laser)', fontsize=18)
    ax4.set_xlabel('Time (s)', fontsize=15)
    ax4.set_ylabel('Electric Field Amplitude', fontsize=15)
    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.grid(True)

    # Zoomed-in PSD plot using PSD from Autocorrelation Function
    ax5 = fig.add_subplot(gs[2, :])

    delta_f = 5e4 
    freq_min = f_central - delta_f / 2
    freq_max = f_central + delta_f / 2

    # Ensure freq_min and freq_max are within the frequency range and greater than zero
    freq_min = max(freq_min, freqs_autocorr_psd[1])  # Use first positive frequency
    freq_max = min(freq_max, freqs_autocorr_psd[-1])

    # Define frequency window
    freq_window = (freqs_autocorr_psd >= freq_min) & (freqs_autocorr_psd <= freq_max)
    freqs_zoom = freqs_autocorr_psd[freq_window]
    psd_zoom = autocorr_laser_psd[freq_window]

    # Convert frequencies to relative frequencies
    freqs_rel = freqs_zoom - f_central

    # Calculate FWHM directly
    psd_max = np.max(psd_zoom)
    half_max = psd_max / 2

    # Find indices where PSD crosses half maximum
    indices_above_half_max = np.where(psd_zoom >= half_max)[0]
    if len(indices_above_half_max) >= 2:
        fwhm = freqs_rel[indices_above_half_max[-1]] - freqs_rel[indices_above_half_max[0]]

        # Plot the PSD
        ax5.plot(freqs_rel, psd_zoom, label='PSD')

        # Draw arrows at half maximum points
        freq_left = freqs_rel[indices_above_half_max[0]]
        freq_right = freqs_rel[indices_above_half_max[-1]]
        y_arrow = psd_zoom[indices_above_half_max[0]]

        # Left arrow pointing right
        ax5.annotate('', xy=(freq_left, y_arrow), xytext=(freq_left - delta_f * 0.01, y_arrow),
                     arrowprops=dict(arrowstyle='->', color='red'))

        # Right arrow pointing left
        ax5.annotate('', xy=(freq_right, y_arrow), xytext=(freq_right + delta_f * 0.01, y_arrow),
                     arrowprops=dict(arrowstyle='->', color='red'))

        # Modify title to include FWHM
        title_text = f'Zoomed PSD around Central Frequency\nEstimated FWHM: {fwhm:.2e} Hz'
    else:
        fwhm = None  # FWHM cannot be determined
        # Plot the PSD
        ax5.plot(freqs_rel, psd_zoom, label='PSD')
        # Modify title to indicate FWHM could not be determined
        title_text = 'Zoomed PSD around Central Frequency'

    ax5.set_title(title_text, fontsize=18)
    ax5.set_xlim(freq_min - f_central, freq_max - f_central)
    # ax5.set_yscale('log')
    ax5.set_xlabel(r'$f - f_0$ (Hz)', fontsize=15)
    ax5.set_ylabel('Power Spectral Density', fontsize=15)
    ax5.tick_params(axis='both', which='major', labelsize=14)
    ax5.grid(True)
    ax5.legend(fontsize=14)

    # Add vertical lines in ax3 at freq_min and freq_max
    ymin, ymax = ax3.get_ylim()
    ax3.vlines([freq_min, freq_max], ymin=ymin, ymax=ymax, colors='gray', linestyles='--', alpha=0.5)

    # Add connection lines from ax3 to ax5
    fig.canvas.draw()  # Necessary to ensure that positions are updated

    # Coordinates in display space
    trans_figure = fig.transFigure.inverted()

    # Get positions in ax3
    x_ax3_min, _ = ax3.transData.transform((freq_min, ymin))
    x_ax3_max, _ = ax3.transData.transform((freq_max, ymin))

    # Get positions in ax5
    x_ax5_min, y_ax5_max = ax5.transData.transform((freq_min - f_central, ax5.get_ylim()[1]))
    x_ax5_max, y_ax5_max = ax5.transData.transform((freq_max - f_central, ax5.get_ylim()[1]))

    # Transform to figure coordinates
    coord1_fig = trans_figure.transform((x_ax3_min, ax3.transAxes.transform((0, 0))[1]))
    coord2_fig = trans_figure.transform((x_ax5_min, y_ax5_max))

    coord3_fig = trans_figure.transform((x_ax3_max, ax3.transAxes.transform((0, 0))[1]))
    coord4_fig = trans_figure.transform((x_ax5_max, y_ax5_max))

    # Create connection patches
    con1 = ConnectionPatch(
        xyA=coord1_fig, coordsA='figure fraction',
        xyB=coord2_fig, coordsB='figure fraction',
        color='gray', linestyle='--', alpha=0.5
    )
    con2 = ConnectionPatch(
        xyA=coord3_fig, coordsA='figure fraction',
        xyB=coord4_fig, coordsB='figure fraction',
        color='gray', linestyle='--', alpha=0.5
    )

    # Add connection patches to the figure
    fig.add_artist(con1)
    fig.add_artist(con2)

    return fig

# Main function to arrange widgets and handle updates
def Examine_Frequency_Noise():
    """
    Sets up the interactive widgets and handles updates to compute and display the PSD.
    """
    # Create output widget
    output = Output()

    # Define layout for widgets
    slider_layout = Layout(width='350px')  # Increased width to prevent label cutoff

    # Define primary widgets
    N_widget = IntText(
        value=25000, 
        description="Num Time Points:", 
        layout=slider_layout
    )
    T_widget = FloatText(
        value=1e-1, 
        description="Total Time (s):", 
        layout=slider_layout
    )
    
    # Initial frequency calculations
    f_min_initial = 1 / T_widget.value
    f_max_initial = N_widget.value / (2 * T_widget.value)
    f_initial = int(abs(f_max_initial - f_min_initial) / 2)
    
    # Define frequency sliders
    f_central_slider = FloatSlider(
        min=f_min_initial, 
        max=f_max_initial, 
        step=f_min_initial, 
        value=f_initial, 
        description="Central Freq (Hz):", 
        continuous_update=False,  # Changed to False for performance
        layout=slider_layout
    )
    f_initial_noise = f_initial - 1e2
    # Define Noise Peak Central Frequency Slider
    noise_peak_central_freq_slider = FloatSlider(
        min=f_min_initial, 
        max=f_max_initial, 
        step=f_min_initial, 
        value=f_initial_noise, 
        description="Noise Peak Freq (Hz):",  # Updated description
        continuous_update=False,  # Changed to False for performance
        layout=slider_layout
    )
    
    def create_psd_widgets(exponent):
        """
        Creates widgets for a specific PSD contribution exponent.

        Parameters:
        - exponent: The exponent value for the PSD contribution.

        Returns:
        - include: Checkbox to include/exclude the PSD contribution.
        - amp: FloatText to set the amplitude of the PSD contribution.
        """
        if exponent == 0:
            include_value = True
            amp_value = 0.1
        else:
            include_value = False
            amp_value = 1 if exponent < 0 else 1e-3  # Default amplitude for exponent >= 0
        include = Checkbox(
            value=include_value, 
            description=f"Include f{superscript(exponent)}"
        )
        amp = FloatText(
            min=1e-8, 
            max=1e10, 
            step=1e-4, 
            value=amp_value, 
            description=f"Amp f{superscript(exponent)}", 
            layout=slider_layout
        )
        return include, amp

    # Helper function to convert exponent to superscript
    def superscript(exponent):
        """
        Converts an integer exponent to its Unicode superscript representation.

        Parameters:
        - exponent: Integer exponent.

        Returns:
        - String with superscript characters.
        """
        superscripts = {
            '-3': '⁻³',
            '-2': '⁻²',
            '-1': '⁻¹',
            '0': '⁰',
            '1': '¹',
            '2': '²'
        }
        return superscripts.get(str(exponent), str(exponent))

    # Create widgets for each exponent without randomness sliders
    exponents = [-3, -2, -1, 0, 1, 2]
    psd_widgets = {}
    for exp in exponents:
        include, amp = create_psd_widgets(exp)
        psd_widgets[exp] = (include, amp)
    
    # Define peak widgets
    include_peak_checkbox = Checkbox(
        value=False, 
        description="Include Noise Peak",
        layout=slider_layout
    )
    peak_amplitude_slider = FloatSlider(
        min=1e-8, 
        max=1e6, 
        step=1e-4, 
        value=1e4, 
        description="Peak Amp.", 
        continuous_update=False,  # Changed to False for performance
        layout=slider_layout
    )
    peak_width_slider = FloatSlider(
        min=1e0, 
        max=1e4, 
        step=1e0, 
        value=1e2, 
        description="Peak Width (Hz)", 
        continuous_update=False,  # Changed to False for performance
        layout=slider_layout
    )
    
    # Define the "Show All" checkbox
    show_all_checkbox = Checkbox(
        value=False,
        description='Show each PSD contribution',
        layout=slider_layout
    )

    # Define the "Show Beta-separation Line" checkbox
    show_beta_line_checkbox = Checkbox(
        value=False,
        description='Show beta-separation line',
        layout=slider_layout
    )

    # Function to update frequency sliders based on N and T
    def update_frequency_sliders(*args):
        """
        Updates the minimum, maximum, and step values of frequency sliders based on N and T.
        """
        new_f_min = 1 / T_widget.value
        new_f_max = N_widget.value / (2 * T_widget.value)
        
        # Update f_central_slider
        f_central_slider.min = new_f_min
        f_central_slider.max = new_f_max
        f_central_slider.step = new_f_min
        f_central_slider.value = np.clip(f_central_slider.value, new_f_min, new_f_max)
        
        # Update noise_peak_central_freq_slider
        noise_peak_central_freq_slider.min = new_f_min
        noise_peak_central_freq_slider.max = new_f_max
        noise_peak_central_freq_slider.step = new_f_min
        noise_peak_central_freq_slider.value = np.clip(noise_peak_central_freq_slider.value, new_f_min, new_f_max)

    # Function to compute and display PSD
    def compute_on_update(*args):
        """
        Computes the PSD based on current widget values and updates the plots.
        """
        with output:
            clear_output(wait=True)
            # Compute the PSD only if necessary parameters have changed
            # Collect the current parameter values
            params = {
                'N': N_widget.value,
                'T': T_widget.value,
                'f_central': f_central_slider.value,
                'include_f3': psd_widgets[-3][0].value,
                'amp_f3': psd_widgets[-3][1].value,
                'include_f2': psd_widgets[-2][0].value,
                'amp_f2': psd_widgets[-2][1].value,
                'include_f1': psd_widgets[-1][0].value,
                'amp_f1': psd_widgets[-1][1].value,
                'include_f0': psd_widgets[0][0].value,
                'amp_f0': psd_widgets[0][1].value,
                'include_fp1': psd_widgets[1][0].value,
                'amp_fp1': psd_widgets[1][1].value,
                'include_fp2': psd_widgets[2][0].value,
                'amp_fp2': psd_widgets[2][1].value,
                'include_peak': include_peak_checkbox.value,
                'peak_location': noise_peak_central_freq_slider.value,
                'peak_amplitude': peak_amplitude_slider.value,
                'peak_width': peak_width_slider.value
            }

            # Check if the parameters have changed
            if 'params' not in computed_data or computed_data['params'] != params:
                # Parameters have changed, recompute the data
                computed_data['params'] = params
                data_dict = compute_psd(**params)
                computed_data['data_dict'] = data_dict
            else:
                # Use cached data
                data_dict = computed_data['data_dict']

            # Plot the results
            fig = plot_all_subplots(
                data_dict, show_all_checkbox.value, show_beta_line_checkbox.value
            )
            plt.show()

    # Function to update plots without recomputing data
    def update_plots_only(*args):
        """
        Updates the plots based on 'show_all' and 'show_beta_line' checkboxes without recomputing data.
        """
        with output:
            clear_output(wait=True)
            if 'data_dict' in computed_data:
                data_dict = computed_data['data_dict']
                fig = plot_all_subplots(
                    data_dict, show_all_checkbox.value, show_beta_line_checkbox.value
                )
                plt.show()
            else:
                compute_on_update()

    # Attach observers to N and T to update frequency sliders
    N_widget.observe(update_frequency_sliders, names='value')
    T_widget.observe(update_frequency_sliders, names='value')
    
    # Attach observers to widgets that require recomputation
    widgets_to_observe_compute = [
        N_widget, T_widget, f_central_slider, noise_peak_central_freq_slider,
        include_peak_checkbox, peak_amplitude_slider, peak_width_slider
    ]
    
    # PSD contribution widgets
    for exp in exponents:
        include, amp = psd_widgets[exp]
        widgets_to_observe_compute.extend([include, amp])
    
    for widget in widgets_to_observe_compute:
        widget.observe(compute_on_update, names='value')
    
    # Attach observers to 'show_all' and 'show_beta_line' checkboxes to update plots only
    show_all_checkbox.observe(update_plots_only, names='value')
    show_beta_line_checkbox.observe(update_plots_only, names='value')
    
    # Arrange PSD contribution widgets
    psd_contrib_boxes = []
    for exp in exponents:
        include, amp = psd_widgets[exp]
        box = HBox([include, amp])
        psd_contrib_boxes.append(box)
    
    # Arrange all controls in a VBox with noise_peak_central_freq_slider added to the last HBox
    controls_box = VBox([
        HBox([N_widget, T_widget]),
        HBox([f_central_slider]),  # Only Central Freq Slider here
        VBox(psd_contrib_boxes),
        VBox([
            include_peak_checkbox,
            HBox([peak_amplitude_slider, peak_width_slider, noise_peak_central_freq_slider]),
            show_all_checkbox,  # Moved checkboxes to the bottom
            show_beta_line_checkbox  # Added Show Beta Line Checkbox
        ])
    ])
    
    # Display the widgets and output area
    display(controls_box)
    display(output)
    
    # Perform initial computation
    compute_on_update()

# Initialize the interactive frequency noise analyzer
Examine_Frequency_Noise()

