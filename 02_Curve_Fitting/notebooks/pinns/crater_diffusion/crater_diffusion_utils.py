import numpy as np
from matplotlib import pyplot as plt    

'''
This module provides functions to calculate the initial topography of craters based on various empirical equations from different studies.

Initial Topography Functions:
    get_crater_dD(D):
        Calculate the depth to diameter ratio for a crater based on its diameter using empirical data from Fasset et al., 2022.

    h_of_r_normal_yang(r, D):
        Calculate the height at a given radial distance for a 'normal' simple crater based on the equations from Yang et al., 2021.
        Valid for craters with diameters between 40 and 100 meters.

    h_of_r_flat_bottomed_yang(r, D):
        Calculate the height at a given radial distance for a 'flat-bottomed' crater based on the equations from Yang et al., 2021.
        Valid for craters with diameters between 40 and 150 meters.

    h_of_r_fasset2014(r, D):
        Calculate the normalized crater depth at a given radial distance based on the equations from Fasset et al., 2014.

    h_of_r_fasset2022(r, D):
        Calculate the height at a given radial distance for a crater based on the equations from Fasset et al., 2022.

'''


# function for depth to diameter ratios from fasset et al., 2022
def get_crater_dD(D):
    """diameter in meters"""
    if D <= 10:
        return 0.1
    elif 10 < D <= 40:
        return 0.11
    elif 40 < D <= 100:
        return 0.13
    elif 100 < D <= 200:
        return 0.15
    elif 200 < D <= 400:
        return 0.17
    else:
        return 0.21


# ---------------------------
# initial topography equations
# ---------------------------

# equation for a 'normal' simple crater from Yang et al.:  http://dx.doi.org/10.1029/2021GL095537
# valid for 40 < D < 100 m
def h_of_r_normal_yang(r, D):
    '''
    r values should be normalized to the crater radius
    D is in meters
    '''

    # empirical constants
    B = 5.8270
    A = -2.8567

    # Diameter-dependent constants
    d0 = 0.114 * D**-0.002
    h_r = 0.02513 * D**-0.0757
    alpha = -3.1906
    C = d0 * ((np.exp(A) + 1) / (np.exp(B) - 1))

    r = np.abs(r)  # Ensure r is positive
    result = np.zeros_like(r)  # Initialize the result array

    # r <= 1 branch
    mask1 = r <= 1
    result[mask1] = C * (
        (np.exp(B * r[mask1]) - np.exp(B)) / (1 + np.exp(A + B * r[mask1]))
    )

    # r > 1 branch
    mask2 = r > 1
    result[mask2] = h_r * (r[mask2] ** alpha - 1)

    return result


import numpy as np


# equation for a 'flat bottomed' crater from Yang et al.:  http://dx.doi.org/10.1029/2021GL095537
# valid for 40 < D < 150 m
def h_of_r_flat_bottomed_yang(r, D):
    """
    Compute h(r) for flat-bottomed topography based on the given equations.

    Parameters:
        r (float or np.ndarray): Radial distance (normalized to crater radius).
        D (float): Diameter of the crater (km).

    Returns:
        h (float or np.ndarray): Height at the given radial distance.
    """
    r = np.abs(r)

    # Constants
    a = -2.6003
    b = 5.8783

    # Compute rb and C
    rb = 0.091 * D**0.208
    d0 = 0.114 * D**-0.002
    h_r = 0.02513 * D**-0.0757
    alpha = -3.1906
    C = d0 * (np.exp(a) + 1) / (np.exp(b) - 1)

    r0 = (r - rb) / (1 - rb)

    # Initialize h
    h = np.zeros_like(r)

    # Define conditions
    condition1 = r <= rb
    condition2 = (rb < r) & (r <= 1)
    condition3 = 1 < r

    # Compute h(r) for each condition
    h[condition1] = -d0
    h[condition2] = (
        C
        * (np.exp(b * r0[condition2]) - np.exp(b))
        / (1 + np.exp(a + b * r0[condition2]))
    )

    # Compute r0 for condition3
    h[condition3] = h_r * (r[condition3] ** alpha - 1)

    return h


# initial crater profile from fasset et al., 2014: https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2014JE004698
def h_of_r_fasset2014(r, D):
    """
    Calculate the normalized crater depth h(r)/D at radial distance r.

    Parameters:
    r: float or numpy array
        Radial distance from crater center (m)
    D: float
        Crater diameter (m)

    Returns:
    float or numpy array
        Normalized crater depth h(r)/D
    """
    # Convert input to numpy array if it isn't already
    r = np.abs(r)  # Ensure r is positive
    r = np.asarray(r)

    R = D / 2  # Crater radius

    # Initialize output array with zeros
    h_D = np.zeros_like(r, dtype=float)

    # Central Flat Floor: r ≤ 0.2R
    mask_central = r <= 0.2 * R
    h_D[mask_central] = -0.181

    # Interior: 0.2R < r < 0.98R
    mask_interior = (r > 0.2 * R) & (r < 0.98 * R)
    h_D[mask_interior] = (
        -0.229
        + 0.228 * (r[mask_interior] / R)
        + 0.083 * (r[mask_interior] / R) ** 2
        - 0.039 * (r[mask_interior] / R) ** 3
    )

    # Rim and Exterior: 0.98R < r < 1.5R
    mask_rim = (r >= 0.98 * R) & (r < 1.5 * R)
    h_D[mask_rim] = (
        0.188
        - 0.187 * (r[mask_rim] / R)
        + 0.018 * (r[mask_rim] / R) ** 2
        + 0.015 * (r[mask_rim] / R) ** 3
    )

    return h_D


# alternate topography equation from Fasset et al.: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022JE007510:
def h_of_r_fasset2022(r, D):
    """
    Calculate the height at a given radial distance for a crater based on the equations from Fasset et al., 2022.
    
    Parameters:
    r: float or numpy array
        Radial distance from the crater center (m)
    D: float
        Crater diameter (m)

    Returns:
    float or numpy array
        Height at the given radial distance (m)
    """

    # Diameter-dependent depth to diameter ratio
    dD = get_crater_dD(D)
    R = D / 2  # Crater radius

    r = np.abs(r)  # Ensure r is positive
    result = np.zeros_like(r)  # Initialize the result array

    # R < 0.98, where Z ≤ –(d/D)initial − 0.0368
    mask2 = r <= 0.98 * R
    result[mask2] = (
        -0.229 + 0.227 * (r[mask2] / R) + 0.083 * (r[mask2] / R) ** 2 - 0.394**3
    )

    mask3 = (r > 0.98 * R) & (r < 1.02 * R)
    result[mask3] = 0.0368

    mask4 = (r >= 1.02 * R) & (r < 1.5 * R)
    result[mask4] = (
        0.188
        - 0.187 * (r[mask4] / R)
        + 0.018 * (r[mask4] / R) ** 2
        + 0.015 * (r[mask4] / R) ** 3
    )

    result = np.where(result <= -(dD - 0.0368), -(dD - 0.0368), result)

    return result


# ---------------------------
# plot functions

def plot_crater_evolution(h_time, x, y, dt_m, kappa, plot_times, title="Crater Evolution"):
    '''
    Plot the evolution of a crater over time.
    
    Parameters:
    h_time: ndarray
        Array of height values over time.
    x: ndarray
        Array of x-coordinates.
    y: ndarray
        Array of y-coordinates.
    dt_m: float
        Time step size in Myr.
    kappa: float
        Diffusivity. m^2/Myr
    plot_times: list
        List of time indices to plot.
    title: str
        Title for the plot.

    Returns:
    fig: matplotlib.figure.Figure
        The figure object.
    '''
    # Create figure with two subplots: one for map views and one for cross section
    n_times = len(plot_times)
    n_cols = 3
    n_rows = (n_times + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 5*(n_rows + 1)))
    
    # Create subplot grid - maps will take up first n_rows, cross section will take bottom row
    gs = plt.GridSpec(n_rows + 1, n_cols, height_ratios=[1]*n_rows + [0.8])
    
    # Plot map views
    axes_maps = []
    vmin = np.min(h_time[plot_times])
    vmax = np.max(h_time[plot_times])

    # make grid from x, y
    X, Y = np.meshgrid(x, y)
    
    for i, time_idx in enumerate(plot_times):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        axes_maps.append(ax)
        kt = dt_m * time_idx * kappa
        
        im = ax.imshow(h_time[time_idx], 
                      extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower',
                      cmap='gist_earth',
                      vmin=vmin,
                      vmax=vmax)
        ax.set_title(f'T = {dt_m * time_idx:.0f} Myr')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        fig.colorbar(im, ax=ax, label='Height (m)')
        
        # Add contour lines
        CS = ax.contour(X, Y, h_time[time_idx], 
                       levels=np.linspace(vmin, vmax, 10),
                       colors='white',
                       alpha=0.3,
                       linewidths=0.5)
        ax.clabel(CS, inline=True, fontsize=8, fmt='%1.1f')
        
        # Add line showing cross section location
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Add kt value as text in the upper right
        ax.text(0.95, 0.95, f'kt = {kt:.2f}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    # Remove any empty map subplots
    for i in range(len(plot_times), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.add_subplot(gs[row, col]).remove()
    
    # Add cross section plot spanning full width
    ax_cross = fig.add_subplot(gs[-1, :])
    # Calculate radial distances from the center
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Define radial bins
    r_bins = np.linspace(0, np.max(R), len(x) // 2)
    r_bin_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    # Generate colors from viridis colormap
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(plot_times)))
    
    # Plot radial average profile for each time step
    for idx, time_idx in enumerate(plot_times):
        h_time_step = h_time[time_idx]
        radial_avg = np.zeros_like(r_bin_centers)
        
        for i in range(len(r_bin_centers)):
            mask = (R >= r_bins[i]) & (R < r_bins[i + 1])
            radial_avg[i] = np.mean(h_time_step[mask])
        
        # Plot mirrored profile as a single line with color from viridis colormap
        ax_cross.plot(np.concatenate([-r_bin_centers[::-1], r_bin_centers]), 
                      np.concatenate([radial_avg[::-1], radial_avg]), 
                      label=f'T = {dt_m * time_idx:.0f} Myr', color=colors[idx])
    
    ax_cross.set_xlabel('Radial Distance (m)')
    ax_cross.set_ylabel('Height (m)')
    ax_cross.set_title('Radial Average Profile')
    ax_cross.legend(ncol=2)
    ax_cross.grid(True, alpha=0.3)

    plt.suptitle(title, y=0.95)
    plt.tight_layout()
    return fig


def plot_crater_times(h_preds, x, y, kappa, plot_times, title="Crater Evolution"):
    """
    slight variation on the plot_crater_evolution function for output from PINN

    Plot the evolution of a crater over time.

    Parameters:
    h_preds: ndarray
        Array of height values over time.
    x: ndarray
        Array of x-coordinates.
    y: ndarray
        Array of y-coordinates.
    plot_times: list
        List of time indices to plot.
    title: str
        Title for the plot.

    """
    # Create figure with two subplots: one for map views and one for cross section
    n_times = len(plot_times)
    n_cols = 3
    n_rows = (n_times + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 5*(n_rows + 1)))
    
    # Create subplot grid - maps will take up first n_rows, cross section will take bottom row
    gs = plt.GridSpec(n_rows + 1, n_cols, height_ratios=[1]*n_rows + [0.8])
    
    # Plot map views
    axes_maps = []
    vmin = np.min(h_preds)
    vmax = np.max(h_preds)

    # make grid from x, y
    X, Y = np.meshgrid(x, y)
    
    for i, time_idx in enumerate(plot_times):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        axes_maps.append(ax)

        kt = time_idx * kappa
        
        im = ax.imshow(h_preds[i], 
                      extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower',
                      cmap='gist_earth',
                      vmin=vmin,
                      vmax=vmax)
        ax.set_title(f'T = {time_idx:.0f} Myr')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        fig.colorbar(im, ax=ax, label='Height (m)')
        
        # Add contour lines
        CS = ax.contour(X, Y, h_preds[i], 
                       levels=np.linspace(vmin, vmax, 10),
                       colors='white',
                       alpha=0.3,
                       linewidths=0.5)
        ax.clabel(CS, inline=True, fontsize=8, fmt='%1.1f')
        
        # Add line showing cross section location
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Add kt value as text in the upper right
        ax.text(0.95, 0.95, f'kt = {kt:.2f}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    # Remove any empty map subplots
    for i in range(len(plot_times), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.add_subplot(gs[row, col]).remove()
    
    # Add cross section plot spanning full width
    ax_cross = fig.add_subplot(gs[-1, :])
    center_idx = len(y) // 2  # Index for y = 0
    
    # Plot cross section for each time step
    for i, time_idx in enumerate(plot_times):
        ax_cross.plot(x, h_preds[i][center_idx, :], 
                     label=f'T = {time_idx:.0f} Myr')
    
    ax_cross.set_xlabel('Distance (m)')
    ax_cross.set_ylabel('Height (m)')
    ax_cross.set_title('Cross Section (y = 0)')
    ax_cross.legend(ncol=2)
    ax_cross.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=0.95)
    plt.tight_layout()
    return fig

# ---------------------------
# data prep functions
import numpy as np

import numpy as np

def prepare_topo_training_data(H, T, test_split=0.2, random_state=42):
    """
    Prepare topographic time series data for training by creating pairs of height data
    at different time steps, along with their time differences.
    
    Parameters:
    -----------
    H : numpy.ndarray
        3D array of height values with shape (t, x, y) where:
        - t is the number of time steps
        - x, y are the spatial dimensions
    T : numpy.ndarray
        1D array of time values corresponding to each time step
    test_split : float, optional
        Fraction of data to reserve for testing (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    h_train : numpy.ndarray
        Array of initial height configurations for training
    dt_train : numpy.ndarray
        Array of time differences between initial and target states for training
    h_test : numpy.ndarray
        Array of initial height configurations for testing
    dt_test : numpy.ndarray
        Array of time differences between initial and target states for testing
    """
    # Get dimensions
    n_times, n_x, n_y = H.shape
    
    # Initialize lists to store pairs
    h_initial = []
    h_target = []
    dt = []
    
    # Generate pairs: for each time step, pair with all future time steps
    for i in range(n_times-1):  # -1 because we need at least one future step
        for j in range(i+1, n_times):
            h_initial.append(H[i])
            h_target.append(H[j])
            dt.append(T[j] - T[i])  # Calculate time difference
    
    # Convert to numpy arrays
    h_initial = np.array(h_initial)
    h_target = np.array(h_target)
    dt = np.array(dt)
    
    # Get number of total pairs
    n_pairs = len(h_initial)
    
    # Create random indices for train/test split
    np.random.seed(random_state)
    indices = np.random.permutation(n_pairs)
    split_idx = int(n_pairs * (1 - test_split))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Split into train and test sets
    h_train = h_initial[train_indices]
    h_target_train = h_target[train_indices]
    dt_train = dt[train_indices]
    h_test = h_initial[test_indices]
    h_target_test = h_target[test_indices]
    dt_test = dt[test_indices]
    
    return h_train, h_target_train, dt_train, h_test, h_target_test, dt_test