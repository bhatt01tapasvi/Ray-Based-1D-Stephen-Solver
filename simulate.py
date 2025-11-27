#!/usr/bin/env python3
"""
simulate.py

1D multilayer transient heat conduction with PCM phase change (enthalpy method).
Produces MF(t) (melt fraction) vs time and saves results to `results/`.

Simplifying assumptions:
- Planar 1D stack (normal rays are identical) so single-ray represents the whole box.
- Layers (from outside to inside): cardboard (4 mm), VIP (20 mm), EPS wall (2 mm), PCM cavity (16 mm).
- Ambient temperature is a sinusoidal day/night profile typical for Singapore.
- Convective BC at outside surface, adiabatic at inner side (approx inside air/payload ignored).

Run: python simulate.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Output folder
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# Geometry (meters)
area = 0.25 * 0.25  # 250 mm x 250 mm footprint
L_cardboard = 0.004
L_vip = 0.020
L_eps_wall = 0.002
L_pcm = 0.016
L_total = L_cardboard + L_vip + L_eps_wall + L_pcm

# Discretization
N = 200  # increased nodes for better resolution (especially in PCM layer)
dx = L_total / N

# Map each node to a material
xs = np.linspace(dx/2, L_total - dx/2, N)

def layer_at(x):
    if x < L_cardboard:
        return 'cardboard'
    if x < L_cardboard + L_vip:
        return 'vip'
    if x < L_cardboard + L_vip + L_eps_wall:
        return 'eps'
    return 'pcm'

materials = np.array([layer_at(x) for x in xs])

# Material properties (typical values)
props = {
    'cardboard': {'k':0.05, 'rho':300, 'cp':1200},
    'vip': {'k':0.005, 'rho':50, 'cp':100},
    'eps': {'k':0.03, 'rho':30, 'cp':1400},
    # PCM (ice/water) properties. For enthalpy method we specify solid/liquid values.
    'pcm': {
        'k_s':2.2, 'k_l':0.6,
        'rho':917, 'cp_s':2050, 'cp_l':4180,
        'Tm':0.0, 'L':334000.0
    }
}

# Initial condition: all -20 degC
T0 = -20.0

# Ambient (Singapore-like) sinusoidal profile (degC)
T_amb_mean = 30.0
T_amb_amp = 4.0  # swings +/- 4 degC
period = 24*3600.0

# Convective coefficient outside
h_out = 10.0

# Time parameters
t_total = 7*24*3600  # 7 days
dt = 120.0  # 120 s time-step (reduced for better accuracy during phase change)
nt = int(t_total / dt)

times = np.linspace(0, t_total, nt+1)

# Initialize temperature field
T = np.ones(N) * T0

# Precompute properties arrays per node
k_arr = np.zeros(N)
rho_arr = np.zeros(N)
cp_arr = np.zeros(N)
is_pcm = np.zeros(N, dtype=bool)
for i, m in enumerate(materials):
    if m == 'pcm':
        p = props['pcm']
        k_arr[i] = p['k_s']
        rho_arr[i] = p['rho']
        cp_arr[i] = p['cp_s']
        is_pcm[i] = True
    else:
        p = props[m]
        k_arr[i] = p['k']
        rho_arr[i] = p['rho']
        cp_arr[i] = p['cp']

# total PCM mass
pcm_vol = np.sum(is_pcm) * dx * area
pcm_mass = pcm_vol * props['pcm']['rho']

# mushy zone parameters - reduced for sharper phase change behavior
dT_mushy = 0.5  # Narrower mushy zone (0.5°C instead of 5°C) for more realistic Stefan behavior
Tm = props['pcm']['Tm']
L = props['pcm']['L']

# helper: ambient temperature at time t
def T_ambient(t):
    return T_amb_mean + T_amb_amp * np.sin(2*pi*t/period)

# TDMA solver for tri-diagonal system
def tdma(a, b, c, d):
    # a: sub-diagonal (len n-1), b: diag (len n), c: super (len n-1), d: rhs
    n = len(b)
    cp = np.empty(n-1)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i-1]*cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1]*dp[i-1]) / denom
    dp[n-1] = (d[n-1] - a[n-2]*dp[n-2]) / (b[n-1] - a[n-2]*cp[n-2])
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

# storage for outputs
mf_t = np.zeros(nt+1)
Tpcm_avg = np.zeros(nt+1)

# compute initial melt fraction and avg pcm T
def compute_outputs(T):
    # enthalpy per kg for PCM nodes
    p = props['pcm']
    ent_pcm = np.zeros_like(T)
    for i in range(N):
        if is_pcm[i]:
            Ti = T[i]
            if Ti < Tm - dT_mushy/2:
                ent_pcm[i] = p['cp_s'] * (Ti - Tm)
            elif Ti > Tm + dT_mushy/2:
                ent_pcm[i] = p['cp_l'] * (Ti - Tm) + p['L']
            else:
                # linear mushy
                frac = (Ti - (Tm - dT_mushy/2)) / dT_mushy
                c_eff = (1-frac)*p['cp_s'] + frac*p['cp_l']
                ent_pcm[i] = c_eff * (Ti - Tm) + frac * p['L']
    melted_mass = 0.0
    pcm_nodes = np.where(is_pcm)[0]
    for i in pcm_nodes:
        # mass in node
        mass = rho_arr[i] * dx * area
        # compute melt fraction in node from enthalpy
        e = ent_pcm[i]
        if e <= 0:
            f = 0.0
        elif e >= L:
            f = 1.0
        else:
            f = e / L
        melted_mass += f * mass
    mf = melted_mass / pcm_mass
    # average pcm temperature
    Tavg = np.mean(T[pcm_nodes]) if len(pcm_nodes)>0 else np.nan
    return mf, Tavg

# initial outputs
mf_t[0], Tpcm_avg[0] = compute_outputs(T)

print("Starting simulation: total time = {:.1f} days, steps = {}".format(t_total/86400.0, nt))

for n in range(1, nt+1):
    t = times[n]
    # build linear system A T_new = b
    # coefficients using semi-implicit: use k at previous T, c_eff at previous T
    k_nodes = k_arr.copy()
    # for PCM switch k between k_s and k_l based on T
    pcm_idx = np.where(is_pcm)[0]
    for i in pcm_idx:
        if T[i] > Tm:
            k_nodes[i] = props['pcm']['k_l']
        else:
            k_nodes[i] = props['pcm']['k_s']

    # effective volumetric heat capacity rho*cp_eff
    rhoCp = np.zeros(N)
    for i in range(N):
        if is_pcm[i]:
            Ti = T[i]
            if Ti < Tm - dT_mushy/2:
                cp_eff = props['pcm']['cp_s']
            elif Ti > Tm + dT_mushy/2:
                cp_eff = props['pcm']['cp_l']
            else:
                frac = (Ti - (Tm - dT_mushy/2)) / dT_mushy
                base = (1-frac)*props['pcm']['cp_s'] + frac*props['pcm']['cp_l']
                # include latent heat over mushy interval but cap to avoid overflow
                cp_eff = base + props['pcm']['L']/dT_mushy
                cp_eff = min(cp_eff, 1e6)
            rhoCp[i] = rho_arr[i] * cp_eff
        else:
            rhoCp[i] = rho_arr[i] * cp_arr[i]

    # build tri-diagonal
    a = np.zeros(N-1)  # sub
    b = np.zeros(N)    # diag
    c = np.zeros(N-1)  # super
    d = np.zeros(N)    # rhs

    # internal interfaces: use k at interface = harmonic mean
    k_interface = np.zeros(N-1)
    for i in range(N-1):
        k_interface[i] = 2.0 * k_nodes[i] * k_nodes[i+1] / (k_nodes[i] + k_nodes[i+1] + 1e-30)

    for i in range(N):
        vol = dx * area
        b_i = rhoCp[i]*vol/dt
        # flux from left and right
        if i > 0:
            A_L = k_interface[i-1]*area/dx
        else:
            A_L = 0.0
        if i < N-1:
            A_R = k_interface[i]*area/dx
        else:
            A_R = 0.0
        b[i] = b_i + A_L + A_R
        d[i] = rhoCp[i]*vol/dt * T[i]
        if i > 0:
            a[i-1] = -A_L
            d[i] += A_L * T[i-1]
        if i < N-1:
            c[i] = -A_R
            d[i] += A_R * T[i+1]

    # boundary condition at outside (node 0): convective to ambient
    T_out = T_ambient(t)
    # modify eq for node 0 to include h_out * area
    A_conv = h_out * area
    b[0] += A_conv
    d[0] += A_conv * T_out

    # inner boundary (last node): assume adiabatic -> no additional term required (A_R=0)

    # Solve
    T_new = tdma(a, b, c, d)
    T = T_new

    # store outputs every step
    if n % 1 == 0:
        mf, tpcm = compute_outputs(T)
        mf_t[n] = mf
        Tpcm_avg[n] = tpcm

        # print a short progress occasionally
        if n % 1440 == 0:  # every day
            print(f"Day {n*dt/86400.0:.1f}: MF={mf*100:.2f}%, Tpcm_avg={tpcm:.2f} C, T_amb={T_out:.2f} C")

# save results
np.savetxt(os.path.join(OUT_DIR, 'time_sec.csv'), times)
np.savetxt(os.path.join(OUT_DIR, 'mf.csv'), mf_t)
np.savetxt(os.path.join(OUT_DIR, 'Tpcm_avg.csv'), Tpcm_avg)

# plot MF vs time (days)
plt.figure(figsize=(6,4))
plt.plot(times/86400.0, mf_t, '-b')
plt.xlabel('Time (days)')
plt.ylabel('PCM Melt Fraction')
plt.ylim(0,1.05)
plt.grid(True)
plt.title('PCM Melt Fraction vs Time')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'mf_vs_time.png'), dpi=200)

# optional: plot average PCM temp
plt.figure(figsize=(6,4))
plt.plot(times/86400.0, Tpcm_avg, '-r')
plt.xlabel('Time (days)')
plt.ylabel('Avg PCM Temperature (C)')
plt.grid(True)
plt.title('Average PCM Temperature vs Time')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Tpcm_avg_vs_time.png'), dpi=200)

print('Simulation finished. Results in', OUT_DIR)
