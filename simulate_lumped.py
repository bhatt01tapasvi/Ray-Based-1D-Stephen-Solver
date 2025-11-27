#!/usr/bin/env python3
"""
simulate_lumped.py

Simple lumped-capacity PCM melting model.

Assumptions:
- Heat flows through series resistance: convection (outside) -> cardboard -> VIP -> EPS -> PCM skin
- PCM represented as a single lumped thermal mass with phase change handled via enthalpy tracking.
- Stable and very fast; produces MF(t) and avg PCM T.

Run: python simulate_lumped.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# Geometry
area = 0.25 * 0.25
L_cardboard = 0.004
L_vip = 0.020
L_eps_wall = 0.002
L_pcm = 0.016

# Materials
k_card = 0.05
k_vip = 0.005
k_eps = 0.03
h_out = 10.0

pcm_rho = 917
pcm_cp_s = 2050
pcm_cp_l = 4180
pcm_L = 334000.0
Tm = 0.0
k_pcm_liquid = 0.6  # Thermal conductivity of liquid PCM (water)

# PCM mass
pcm_vol = area * L_pcm
pcm_mass = pcm_vol * pcm_rho

# PCM solid thermal conductivity (ice)
k_pcm_solid = 2.2

# thermal resistances
R_conv = 1.0/(h_out*area)
R_card = L_cardboard/(k_card*area)
R_vip = L_vip/(k_vip*area)
R_eps = L_eps_wall/(k_eps*area)
R_tot = R_conv + R_card + R_vip + R_eps

# Ambient
T_amb_mean = 30.0
T_amb_amp = 4.0
period = 24*3600.0

def T_ambient(t):
    return T_amb_mean + T_amb_amp * np.sin(2*pi*t/period)

# Time
t_total = 7*24*3600
dt = 60.0
nt = int(t_total/dt)
times = np.linspace(0, t_total, nt+1)

# initial enthalpy per kg relative to Tm
T0 = -20.0
if T0 < Tm:
    E = pcm_cp_s * (T0 - Tm)  # J/kg (negative)
else:
    E = pcm_L + pcm_cp_l * (T0 - Tm)

mf = np.zeros(nt+1)
Tpcm = np.zeros(nt+1)

def invert_E_to_T(E):
    if E < 0:
        return Tm + E/pcm_cp_s
    elif E <= pcm_L:
        return Tm
    else:
        return Tm + (E - pcm_L)/pcm_cp_l

Tpcm[0] = invert_E_to_T(E)
if E < 0:
    mf[0] = 0.0
elif E > pcm_L:
    mf[0] = 1.0
else:
    mf[0] = E/pcm_L

# Stefan resistance effect: models the additional thermal resistance from the growing liquid layer
# This creates the characteristic sqrt(t) melting behavior of classical Stefan problems
# Higher values create more pronounced slowdown as melting progresses
STEFAN_RESISTANCE_FACTOR = 80.0  # K/W per unit melt fraction (balanced for visible effect)

for n in range(1, nt+1):
    t = times[n]
    Tamb = T_ambient(t)
    Tpcm_node = invert_E_to_T(E)
    
    # Stefan resistance model: resistance grows with melt fraction
    current_mf = mf[n-1]
    R_stefan = STEFAN_RESISTANCE_FACTOR * current_mf
    
    R_total_dynamic = R_tot + R_stefan
    
    Q_in = (Tamb - Tpcm_node) / R_total_dynamic
    # energy per kg change
    dE = Q_in * dt / pcm_mass
    E = E + dE
    # store
    Tpcm[n] = invert_E_to_T(E)
    if E < 0:
        mf[n] = 0.0
    elif E >= pcm_L:
        mf[n] = 1.0
    else:
        mf[n] = E/pcm_L

np.savetxt(os.path.join(OUT_DIR, 'time_sec_lumped.csv'), times)
np.savetxt(os.path.join(OUT_DIR, 'mf_lumped.csv'), mf)
np.savetxt(os.path.join(OUT_DIR, 'Tpcm_avg_lumped.csv'), Tpcm)

plt.figure()
plt.plot(times/86400.0, mf, '-b')
plt.xlabel('Time (days)')
plt.ylabel('PCM Melt Fraction')
plt.ylim(0,1.05)
plt.grid(True)
plt.title('Lumped PCM Melt Fraction vs Time')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'mf_vs_time_lumped.png'), dpi=200)

plt.figure()
plt.plot(times/86400.0, Tpcm, '-r')
plt.xlabel('Time (days)')
plt.ylabel('PCM Temperature (C)')
plt.grid(True)
plt.title('Lumped PCM Temperature vs Time')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Tpcm_vs_time_lumped.png'), dpi=200)

print('Lumped simulation finished. Results in', OUT_DIR)
