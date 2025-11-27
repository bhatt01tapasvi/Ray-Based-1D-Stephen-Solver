#!/usr/bin/env python3
"""
1D Stefan Problem Solver for PCM Melting Simulation

This module implements a finite difference solver for the 1D Stefan problem,
which models phase change materials (PCM) during melting. The solver uses
the enthalpy method to handle the moving boundary problem.

Features:
- Enthalpy-based formulation for phase change
- Numerically stable TDMA (Thomas Algorithm) solver
- Both lumped and distributed parameter models
- Non-linear mass fraction curves with proper physics
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# Constants
# ==============================================================================
T_MIN_BOUND = -50.0  # Minimum temperature bound [°C]
T_MAX_BOUND = 150.0  # Maximum temperature bound [°C]

# ==============================================================================
# Default PCM Material Properties (Paraffin RT-42)
# ==============================================================================
DEFAULT_PARAMS = {
    # Thermal properties
    'cp_s': 2000.0,        # Solid specific heat [J/(kg·K)]
    'cp_l': 2200.0,        # Liquid specific heat [J/(kg·K)]
    'k_s': 0.2,            # Solid thermal conductivity [W/(m·K)]
    'k_l': 0.15,           # Liquid thermal conductivity [W/(m·K)]
    'rho': 850.0,          # Density [kg/m³]
    'L_f': 200000.0,       # Latent heat of fusion [J/kg]
    'Tm': 42.0,            # Melting temperature [°C]
    'dT': 2.0,             # Mushy zone half-width [°C]
    
    # Geometry
    'length': 0.05,        # PCM thickness [m]
    'n_nodes': 100,        # Number of spatial nodes
    
    # Simulation parameters
    'dt': 30.0,            # Time step [s] - smaller for stability
    'total_days': 7.0,     # Total simulation time [days]
    
    # Boundary conditions
    'T_init': 25.0,        # Initial temperature [°C]
    'T_hot': 60.0,         # Hot side temperature [°C]
    'T_amb': 25.0,         # Ambient temperature [°C]
    'h_conv': 5.0,         # Convective heat transfer coefficient [W/(m²·K)]
    
    # Solar radiation parameters (optional)
    'use_solar': False,    # Disable solar for cleaner curves
    'I_solar': 500.0,      # Peak solar irradiance [W/m²]
    'alpha_abs': 0.9,      # Absorptivity of the surface
    'A_surface': 0.01,     # Surface area [m²]
}


def harmonic_mean(a, b, eps=1e-10):
    """
    Calculate the harmonic mean of two values.
    
    Used for interface thermal conductivity calculation.
    
    Args:
        a: First value
        b: Second value
        eps: Small value to prevent division by zero
    
    Returns:
        Harmonic mean: 2*a*b / (a + b)
    """
    return 2 * a * b / (a + b + eps)


def create_results_directory(base_dir="results"):
    """Create a timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def solar_irradiance(t, I_max, day_length=12*3600):
    """
    Calculate solar irradiance at time t.
    
    Args:
        t: Time in seconds
        I_max: Maximum solar irradiance [W/m²]
        day_length: Length of day in seconds (default 12 hours)
    
    Returns:
        Solar irradiance [W/m²]
    """
    day_seconds = 24 * 3600
    time_of_day = t % day_seconds
    sunrise = (day_seconds - day_length) / 2
    sunset = sunrise + day_length
    
    if sunrise <= time_of_day <= sunset:
        # Sinusoidal variation during daytime
        phase = np.pi * (time_of_day - sunrise) / day_length
        return I_max * np.sin(phase)
    else:
        return 0.0


def calculate_liquid_fraction(T, Tm, dT):
    """
    Calculate liquid fraction based on temperature with a smooth transition.
    
    Args:
        T: Temperature [°C]
        Tm: Melting temperature [°C]
        dT: Mushy zone half-width [°C]
    
    Returns:
        Liquid fraction (0 to 1)
    """
    if dT <= 0:
        dT = 0.1  # Prevent division by zero
    
    if np.isscalar(T):
        if T < Tm - dT:
            return 0.0
        elif T > Tm + dT:
            return 1.0
        else:
            return 0.5 * (1 + (T - Tm) / dT)
    else:
        f = np.zeros_like(T, dtype=np.float64)
        solid = T < Tm - dT
        liquid = T > Tm + dT
        mushy = ~solid & ~liquid
        
        f[liquid] = 1.0
        f[mushy] = 0.5 * (1 + (T[mushy] - Tm) / dT)
        return f


def calculate_enthalpy(T, p):
    """
    Calculate enthalpy from temperature using the enthalpy method.
    
    Args:
        T: Temperature array [°C]
        p: Parameters dictionary
    
    Returns:
        Enthalpy array [J/kg]
    """
    Tm = p['Tm']
    dT = p['dT']
    cp_s = p['cp_s']
    cp_l = p['cp_l']
    L_f = p['L_f']
    
    n = len(T) if not np.isscalar(T) else 1
    H = np.zeros(n) if n > 1 else 0.0
    
    if np.isscalar(T):
        Ti = float(T)
        if Ti < Tm - dT:
            # Solid phase
            return cp_s * (Ti - (Tm - dT))
        elif Ti > Tm + dT:
            # Liquid phase
            return cp_s * (2 * dT) + L_f + cp_l * (Ti - (Tm + dT))
        else:
            # Mushy zone
            f = 0.5 * (1 + (Ti - Tm) / dT)
            return cp_s * (Ti - (Tm - dT)) + f * L_f
    else:
        for i, Ti in enumerate(T):
            if Ti < Tm - dT:
                # Solid phase
                H[i] = cp_s * (Ti - (Tm - dT))
            elif Ti > Tm + dT:
                # Liquid phase
                H[i] = cp_s * (2 * dT) + L_f + cp_l * (Ti - (Tm + dT))
            else:
                # Mushy zone
                f = 0.5 * (1 + (Ti - Tm) / dT)
                H[i] = cp_s * (Ti - (Tm - dT)) + f * L_f
        return H


def calculate_temperature_from_enthalpy(H, p):
    """
    Calculate temperature from enthalpy (inverse of calculate_enthalpy).
    
    Args:
        H: Enthalpy array [J/kg]
        p: Parameters dictionary
    
    Returns:
        Temperature array [°C]
    """
    Tm = p['Tm']
    dT = p['dT']
    cp_s = p['cp_s']
    cp_l = p['cp_l']
    L_f = p['L_f']
    
    # Reference enthalpies
    H_solid_end = cp_s * (2 * dT)  # Enthalpy at end of solid phase
    H_liquid_start = H_solid_end + L_f  # Enthalpy at start of liquid phase
    
    n = len(H) if not np.isscalar(H) else 1
    T = np.zeros(n) if n > 1 else 0.0
    
    if np.isscalar(H):
        Hi = float(H)
        if Hi < 0:
            # Below reference (solid)
            return (Tm - dT) + Hi / cp_s
        elif Hi < H_solid_end:
            # Solid phase
            return (Tm - dT) + Hi / cp_s
        elif Hi < H_liquid_start:
            # Mushy zone
            f = (Hi - H_solid_end) / L_f if L_f > 0 else 0.5
            return Tm + dT * (2 * f - 1)
        else:
            # Liquid phase
            return (Tm + dT) + (Hi - H_liquid_start) / cp_l
    else:
        for i, Hi in enumerate(H):
            if Hi < 0:
                # Below reference (solid)
                T[i] = (Tm - dT) + Hi / cp_s
            elif Hi < H_solid_end:
                # Solid phase
                T[i] = (Tm - dT) + Hi / cp_s
            elif Hi < H_liquid_start:
                # Mushy zone
                f = (Hi - H_solid_end) / L_f if L_f > 0 else 0.5
                T[i] = Tm + dT * (2 * f - 1)
            else:
                # Liquid phase
                T[i] = (Tm + dT) + (Hi - H_liquid_start) / cp_l
        return T


def effective_thermal_properties(T, p):
    """
    Calculate effective thermal properties based on local liquid fraction.
    
    Args:
        T: Temperature array [°C]
        p: Parameters dictionary
    
    Returns:
        Tuple of (k_eff, cp_eff) arrays
    """
    f = calculate_liquid_fraction(T, p['Tm'], p['dT'])
    
    # Linear interpolation of properties
    k_eff = p['k_s'] * (1 - f) + p['k_l'] * f
    cp_eff = p['cp_s'] * (1 - f) + p['cp_l'] * f
    
    # Add apparent heat capacity in mushy zone
    # This accounts for latent heat absorption
    if not np.isscalar(T):
        in_mushy = (T >= p['Tm'] - p['dT']) & (T <= p['Tm'] + p['dT'])
        if np.any(in_mushy):
            # Apparent cp = base_cp + L_f / (2 * dT)
            cp_app = p['L_f'] / (2 * p['dT'])
            cp_eff[in_mushy] += cp_app
    else:
        if p['Tm'] - p['dT'] <= T <= p['Tm'] + p['dT']:
            cp_app = p['L_f'] / (2 * p['dT'])
            cp_eff += cp_app
    
    return k_eff, cp_eff


def tdma_solver(a, b, c, d):
    """
    Tridiagonal Matrix Algorithm (Thomas Algorithm) solver.
    
    Solves the system of equations:
    a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    
    Args:
        a: Lower diagonal coefficients (a[0] is not used)
        b: Main diagonal coefficients
        c: Upper diagonal coefficients (c[n-1] is not used)
        d: Right-hand side vector
    
    Returns:
        Solution vector x
    """
    n = len(d)
    
    # Create working copies with float64 for precision
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    d = np.array(d, dtype=np.float64)
    
    # Modified coefficients
    cp = np.zeros(n, dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)
    
    # Forward elimination
    # Avoid division by zero
    denom = b[0]
    if abs(denom) < 1e-15:
        denom = 1e-15 if denom >= 0 else -1e-15
    cp[0] = c[0] / denom
    dp[0] = d[0] / denom
    
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        # Numerical stability: prevent division by very small numbers
        if abs(denom) < 1e-15:
            denom = 1e-15 if denom >= 0 else -1e-15
        
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
        
        # Check for numerical issues
        if not np.isfinite(dp[i]):
            # Reset to previous value if overflow
            dp[i] = dp[i-1] if i > 0 else 0.0
    
    # Back substitution
    x = np.zeros(n, dtype=np.float64)
    x[n-1] = dp[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
        
        # Check for numerical issues
        if not np.isfinite(x[i]):
            x[i] = x[i+1]
    
    return x


def simulate_distributed(p=None, verbose=True):
    """
    Run a distributed parameter (1D finite difference) simulation.
    
    This solves the 1D Stefan problem using the enthalpy method with
    an explicit finite difference scheme. The enthalpy formulation
    naturally handles the moving phase boundary and produces the
    characteristic melting curve.
    
    Args:
        p: Parameters dictionary (uses defaults if None)
        verbose: Print progress during simulation
    
    Returns:
        Dictionary containing simulation results
    """
    if p is None:
        p = DEFAULT_PARAMS.copy()
    else:
        # Merge with defaults
        params = DEFAULT_PARAMS.copy()
        params.update(p)
        p = params
    
    # Extract parameters
    n = p['n_nodes']
    L = p['length']
    dx = L / (n - 1)
    dt_output = p['dt']  # Time step for output
    total_time = p['total_days'] * 24 * 3600  # seconds
    n_steps = int(total_time / dt_output)
    
    rho = p['rho']
    k_s = p['k_s']
    k_l = p['k_l']
    cp_s = p['cp_s']
    cp_l = p['cp_l']
    L_f = p['L_f']
    Tm = p['Tm']
    dT = p['dT']
    T_init = p['T_init']
    T_hot = p['T_hot']
    T_amb = p['T_amb']
    h_conv = p['h_conv']
    
    # Calculate maximum stable time step for explicit scheme
    # Stability criterion: alpha * dt / dx^2 <= 0.5
    alpha_max = max(k_s, k_l) / (rho * min(cp_s, cp_l))
    dt_stable = 0.4 * dx**2 / alpha_max  # Use 0.4 for safety margin
    
    # Use sub-stepping if needed
    n_substeps = max(1, int(np.ceil(dt_output / dt_stable)))
    dt = dt_output / n_substeps
    
    if verbose:
        print(f"Starting distributed simulation: total time = {p['total_days']} days, steps = {n_steps}")
        if n_substeps > 1:
            print(f"  Using {n_substeps} substeps per output step for numerical stability")
    
    # Spatial grid
    x = np.linspace(0, L, n)
    
    # Initialize temperature field
    T = np.ones(n) * T_init
    
    # Initialize enthalpy field
    # H = 0 at T = Tm - dT (start of melting)
    H = np.zeros(n)
    for i in range(n):
        if T[i] < Tm - dT:
            H[i] = rho * cp_s * (T[i] - (Tm - dT))
        elif T[i] > Tm + dT:
            H[i] = rho * (cp_s * 2 * dT + L_f + cp_l * (T[i] - (Tm + dT)))
        else:
            f = 0.5 * (1 + (T[i] - Tm) / dT)
            H[i] = rho * (cp_s * (T[i] - (Tm - dT)) + f * L_f)
    
    # Enthalpy thresholds (per unit volume)
    H_solid_end = rho * cp_s * (2 * dT)
    H_melt_end = H_solid_end + rho * L_f
    
    # Storage for results
    times = []
    mass_fractions = []
    avg_temps = []
    interface_positions = []
    T_profiles = []
    
    # Main time loop (over output steps)
    for step in range(n_steps):
        t_output = step * dt_output
        
        # Sub-stepping for numerical stability
        for substep in range(n_substeps):
            t = t_output + substep * dt
            
            # Calculate temperature from enthalpy at each node
            for i in range(n):
                if H[i] < 0:
                    T[i] = (Tm - dT) + H[i] / (rho * cp_s)
                elif H[i] < H_solid_end:
                    T[i] = (Tm - dT) + H[i] / (rho * cp_s)
                elif H[i] < H_melt_end:
                    # In mushy zone - temperature varies linearly
                    f = (H[i] - H_solid_end) / (H_melt_end - H_solid_end)
                    T[i] = Tm - dT + 2 * dT * f
                else:
                    T[i] = (Tm + dT) + (H[i] - H_melt_end) / (rho * cp_l)
            
            # Calculate effective thermal conductivity
            k_eff = np.zeros(n)
            for i in range(n):
                if H[i] < H_solid_end:
                    k_eff[i] = k_s
                elif H[i] > H_melt_end:
                    k_eff[i] = k_l
                else:
                    # Linear interpolation in mushy zone
                    f = (H[i] - H_solid_end) / (H_melt_end - H_solid_end)
                    k_eff[i] = k_s * (1 - f) + k_l * f
            
            # Calculate heat flux and update enthalpy
            H_new = H.copy()
            
            # Interior nodes: explicit update
            for i in range(1, n-1):
                # Interface thermal conductivities (harmonic mean)
                k_e = harmonic_mean(k_eff[i], k_eff[i+1])
                k_w = harmonic_mean(k_eff[i], k_eff[i-1])
                
                # Heat flux (positive = heat flowing in positive x direction)
                q_e = k_e * (T[i+1] - T[i]) / dx
                q_w = k_w * (T[i] - T[i-1]) / dx
                
                # Enthalpy update: dH/dt = d(k dT/dx)/dx = (q_e - q_w)/dx
                dH_dt = (q_e - q_w) / dx
                H_new[i] = H[i] + dH_dt * dt
            
            # Left boundary (x=0): Dirichlet (hot side)
            if p['use_solar']:
                Q_solar = solar_irradiance(t, p['I_solar']) * p['alpha_abs']
                T_left = T_hot + Q_solar / (h_conv + 0.1)
                T_left = min(T_left, 100.0)
            else:
                T_left = T_hot
            
            # Apply Dirichlet BC: set enthalpy based on T_left
            if T_left < Tm - dT:
                H_new[0] = rho * cp_s * (T_left - (Tm - dT))
            elif T_left > Tm + dT:
                H_new[0] = H_melt_end + rho * cp_l * (T_left - (Tm + dT))
            else:
                f = 0.5 * (1 + (T_left - Tm) / dT)
                H_new[0] = rho * cp_s * (T_left - (Tm - dT)) + f * rho * L_f
            
            # Right boundary (x=L): Convective (Newton cooling)
            k_n = k_eff[n-1]
            # Heat conducted into boundary from interior
            q_cond_in = k_n * (T[n-2] - T[n-1]) / dx
            # Heat lost to ambient
            q_conv_out = h_conv * (T[n-1] - T_amb)
            
            # Net heat flux into boundary node
            dH_dt = (q_cond_in - q_conv_out) / (dx / 2)  # Half control volume
            H_new[n-1] = H[n-1] + dH_dt * dt
            
            # Update enthalpy
            H = H_new.copy()
        
        # After sub-stepping, calculate results for output
        # Calculate liquid fraction at each node
        f_local = np.zeros(n)
        for i in range(n):
            if H[i] <= H_solid_end:
                f_local[i] = 0.0
            elif H[i] >= H_melt_end:
                f_local[i] = 1.0
            else:
                f_local[i] = (H[i] - H_solid_end) / (H_melt_end - H_solid_end)
        
        mf = np.mean(f_local)
        
        # Update temperature from enthalpy for output
        for i in range(n):
            if H[i] < 0:
                T[i] = (Tm - dT) + H[i] / (rho * cp_s)
            elif H[i] < H_solid_end:
                T[i] = (Tm - dT) + H[i] / (rho * cp_s)
            elif H[i] < H_melt_end:
                f = (H[i] - H_solid_end) / (H_melt_end - H_solid_end)
                T[i] = Tm - dT + 2 * dT * f
            else:
                T[i] = (Tm + dT) + (H[i] - H_melt_end) / (rho * cp_l)
        
        # Ensure temperature bounds
        T = np.clip(T, T_MIN_BOUND, T_MAX_BOUND)
        
        # Find interface position (where liquid fraction ≈ 0.5)
        interface_idx = np.argmin(np.abs(f_local - 0.5))
        interface_pos = x[interface_idx]
        
        # Store results at regular intervals
        if step % max(1, n_steps // 1000) == 0:
            times.append(t_output)
            mass_fractions.append(mf)
            avg_temps.append(np.mean(T))
            interface_positions.append(interface_pos)
            
            if step % max(1, n_steps // 10) == 0:
                T_profiles.append(T.copy())
        
        # Progress report
        if verbose and step % (n_steps // 10) == 0 and step > 0:
            day = t_output / (24 * 3600)
            print(f"Day {day:.1f}: MF={mf*100:.2f}%, Tpcm_avg={np.mean(T):.2f} C, T_amb={T_amb:.2f} C")
    
    if verbose:
        print("Distributed simulation finished. Results in results")
    
    return {
        'times': np.array(times),
        'mass_fractions': np.array(mass_fractions),
        'avg_temps': np.array(avg_temps),
        'interface_positions': np.array(interface_positions),
        'T_profiles': T_profiles,
        'x': x,
        'params': p,
        'type': 'distributed'
    }


def simulate_lumped(p=None, verbose=True):
    """
    Run a lumped parameter simulation for comparison.
    
    The lumped model uses a simplified approach where the PCM is treated
    as a single thermal mass. This model tracks enthalpy to properly handle
    phase change and produces the characteristic Stefan problem curve.
    
    The key physics:
    - Initially high melting rate when T_hot - T_interface is large
    - Rate decreases as more material melts (thermal resistance increases)
    - Approaches zero asymptotically as system reaches thermal equilibrium
    
    Args:
        p: Parameters dictionary (uses defaults if None)
        verbose: Print progress during simulation
    
    Returns:
        Dictionary containing simulation results
    """
    if p is None:
        p = DEFAULT_PARAMS.copy()
    else:
        params = DEFAULT_PARAMS.copy()
        params.update(p)
        p = params
    
    # Extract parameters
    dt = p['dt']
    total_time = p['total_days'] * 24 * 3600  # seconds
    n_steps = int(total_time / dt)
    
    rho = p['rho']
    L = p['length']
    A = p['A_surface']
    V = L * A  # Volume
    Tm = p['Tm']
    dT = p['dT']
    L_f = p['L_f']
    k_s = p['k_s']
    k_l = p['k_l']
    cp_s = p['cp_s']
    cp_l = p['cp_l']
    T_init = p['T_init']
    T_hot = p['T_hot']
    T_amb = p['T_amb']
    h_conv = p['h_conv']
    
    if verbose:
        print(f"Starting lumped simulation: total time = {p['total_days']} days, steps = {n_steps}")
    
    # Initialize state variables
    m = rho * V  # Total mass
    T = T_init   # Current temperature
    mf = 0.0     # Mass fraction melted
    
    # Storage for results
    times = []
    mass_fractions = []
    temps = []
    
    # Phase tracking
    phase = 'solid'  # 'solid', 'mushy', or 'liquid'
    
    for step in range(n_steps):
        t = step * dt
        
        # Calculate heat input
        if p['use_solar']:
            Q_solar = solar_irradiance(t, p['I_solar']) * p['alpha_abs'] * A
        else:
            Q_solar = 0.0
        
        # Heat conduction from hot side (simplified)
        # Use effective thermal conductivity and average distance
        k_eff = k_s * (1 - mf) + k_l * mf
        Q_in = k_eff * A * (T_hot - T) / (L / 2)
        
        # Heat loss to ambient
        Q_out = h_conv * A * (T - T_amb)
        
        # Net heat
        Q_net = Q_solar + Q_in - Q_out
        
        # Ensure numerical stability
        if not np.isfinite(Q_net):
            Q_net = 0.0
        
        # Update state based on phase
        if phase == 'solid':
            # Solid phase - sensible heating
            dT_dt = Q_net / (m * cp_s)
            T_new = T + dT_dt * dt
            
            if T_new >= Tm - dT:
                # Enter mushy zone
                excess_energy = m * cp_s * (T_new - (Tm - dT))
                T = Tm - dT
                mf = excess_energy / (m * L_f)
                mf = np.clip(mf, 0, 1)
                phase = 'mushy'
            else:
                T = T_new
            
        elif phase == 'mushy':
            # Phase change - absorb latent heat
            dmf_dt = Q_net / (m * L_f)
            mf_new = mf + dmf_dt * dt
            
            if mf_new >= 1.0:
                # Complete melting, enter liquid phase
                mf = 1.0
                T = Tm + dT
                phase = 'liquid'
            elif mf_new <= 0.0:
                # Re-solidification
                mf = 0.0
                T = Tm - dT
                phase = 'solid'
            else:
                mf = mf_new
                # Temperature in mushy zone varies with mf
                T = Tm - dT + (2 * dT) * mf
            
        else:  # liquid
            # Liquid phase - sensible heating
            dT_dt = Q_net / (m * cp_l)
            T_new = T + dT_dt * dt
            
            if T_new <= Tm + dT:
                # Re-enter mushy zone (cooling)
                T = Tm + dT
                mf = 1.0
                phase = 'mushy'
            else:
                T = T_new
        
        # Bounds checking
        T = np.clip(T, T_MIN_BOUND, T_MAX_BOUND)
        mf = np.clip(mf, 0, 1)
        
        # Store results at regular intervals
        if step % max(1, n_steps // 1000) == 0:
            times.append(t)
            mass_fractions.append(mf)
            temps.append(T)
        
        # Progress report
        if verbose and step % (n_steps // 10) == 0 and step > 0:
            day = t / (24 * 3600)
            print(f"Day {day:.1f}: MF={mf*100:.2f}%, Tpcm_avg={T:.2f} C, T_amb={T_amb:.2f} C")
    
    if verbose:
        print("Lumped simulation finished. Results in results")
    
    return {
        'times': np.array(times),
        'mass_fractions': np.array(mass_fractions),
        'avg_temps': np.array(temps),
        'params': p,
        'type': 'lumped'
    }


def plot_results(results, save_dir=None):
    """
    Plot simulation results.
    
    Args:
        results: Dictionary of simulation results
        save_dir: Directory to save plots (None for display only)
    """
    times_hours = results['times'] / 3600
    times_days = results['times'] / (24 * 3600)
    mf = results['mass_fractions'] * 100  # Convert to percentage
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Mass Fraction vs Time
    ax1 = axes[0, 0]
    ax1.plot(times_hours, mf, 'b-', linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Mass Fraction (%)')
    ax1.set_title('Mass Fraction vs Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Plot 2: Average Temperature vs Time
    ax2 = axes[0, 1]
    ax2.plot(times_hours, results['avg_temps'], 'r-', linewidth=2)
    ax2.axhline(y=results['params']['Tm'], color='k', linestyle='--', 
                label=f'Tm = {results["params"]["Tm"]} °C')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Average Temperature (°C)')
    ax2.set_title('Average PCM Temperature vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rate of Change of Mass Fraction
    ax3 = axes[1, 0]
    if len(mf) > 1:
        dmf_dt = np.gradient(mf, times_hours)
        ax3.plot(times_hours, dmf_dt, 'g-', linewidth=2)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Rate of Change (%/hour)')
        ax3.set_title('Rate of Melting vs Time')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Temperature profiles (for distributed model)
    ax4 = axes[1, 1]
    if 'T_profiles' in results and len(results['T_profiles']) > 0:
        x_mm = results['x'] * 1000  # Convert to mm
        colors = plt.cm.viridis(np.linspace(0, 1, len(results['T_profiles'])))
        for i, T_prof in enumerate(results['T_profiles']):
            time_label = f't = {i * results["params"]["total_days"] / len(results["T_profiles"]):.1f} days'
            ax4.plot(x_mm, T_prof, color=colors[i], linewidth=1.5, label=time_label)
        ax4.set_xlabel('Position (mm)')
        ax4.set_ylabel('Temperature (°C)')
        ax4.set_title('Temperature Profiles')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Temperature profiles\nnot available for\nlumped model', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Temperature Profiles (N/A for lumped)')
    
    plt.tight_layout()
    
    if save_dir:
        plot_file = os.path.join(save_dir, f'results_{results["type"]}.png')
        plt.savefig(plot_file, dpi=150)
        print(f"Plot saved to {plot_file}")
    
    plt.close(fig)
    
    return fig


def compare_models(params=None, save_dir=None):
    """
    Run both lumped and distributed simulations and compare results.
    
    Args:
        params: Optional parameter overrides
        save_dir: Directory to save results
    
    Returns:
        Tuple of (lumped_results, distributed_results)
    """
    if save_dir is None:
        save_dir = create_results_directory()
    
    print("=" * 60)
    print("LUMPED MODEL SIMULATION")
    print("=" * 60)
    lumped = simulate_lumped(params, verbose=True)
    plot_results(lumped, save_dir)
    
    print("\n" + "=" * 60)
    print("DISTRIBUTED MODEL SIMULATION")
    print("=" * 60)
    distributed = simulate_distributed(params, verbose=True)
    plot_results(distributed, save_dir)
    
    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    times_hours_l = lumped['times'] / 3600
    times_hours_d = distributed['times'] / 3600
    
    # Mass fraction comparison
    ax1 = axes[0]
    ax1.plot(times_hours_l, lumped['mass_fractions'] * 100, 'b-', 
             linewidth=2, label='Lumped Model')
    ax1.plot(times_hours_d, distributed['mass_fractions'] * 100, 'r--', 
             linewidth=2, label='Distributed Model')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Mass Fraction (%)')
    ax1.set_title('Mass Fraction Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Temperature comparison
    ax2 = axes[1]
    ax2.plot(times_hours_l, lumped['avg_temps'], 'b-', 
             linewidth=2, label='Lumped Model')
    ax2.plot(times_hours_d, distributed['avg_temps'], 'r--', 
             linewidth=2, label='Distributed Model')
    ax2.axhline(y=lumped['params']['Tm'], color='k', linestyle=':', 
                label=f'Tm = {lumped["params"]["Tm"]} °C')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Average Temperature (°C)')
    ax2.set_title('Temperature Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    comparison_file = os.path.join(save_dir, 'comparison.png')
    plt.savefig(comparison_file, dpi=150)
    print(f"\nComparison plot saved to {comparison_file}")
    plt.close(fig)
    
    # Save numerical results to CSV
    csv_file = os.path.join(save_dir, 'results.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time (s)', 'Time (hours)', 'MF_lumped (%)', 'MF_distributed (%)',
                        'T_lumped (C)', 'T_distributed (C)'])
        
        # Use minimum length for comparison
        n = min(len(lumped['times']), len(distributed['times']))
        for i in range(n):
            writer.writerow([
                lumped['times'][i],
                lumped['times'][i] / 3600,
                lumped['mass_fractions'][i] * 100,
                distributed['mass_fractions'][i] * 100,
                lumped['avg_temps'][i],
                distributed['avg_temps'][i]
            ])
    
    print(f"Results saved to {csv_file}")
    
    return lumped, distributed


def main():
    """Main entry point for the simulation."""
    # Create results directory
    results_dir = create_results_directory()
    
    # Run comparison of both models
    compare_models(save_dir=results_dir)
    
    print(f"\nAll results saved in: {results_dir}")


if __name__ == "__main__":
    main()
