# Ray-Based-1D-Stefan-Solver

A 1D Stefan problem solver for Phase Change Material (PCM) melting simulations. This implementation uses the enthalpy method to handle the moving phase boundary and produces the characteristic melting curve with:
- Initially high melting rate
- Rapidly decreasing rate as melting progresses  
- Asymptotic approach to equilibrium

## Features

- **Lumped Parameter Model**: Simplified model treating PCM as a single thermal mass
- **Distributed Parameter Model**: Full 1D finite difference solution with enthalpy formulation
- **Numerically Stable**: Automatic sub-stepping for explicit scheme stability
- **Flexible Parameters**: Customizable material properties, geometry, and boundary conditions
- **Visualization**: Automatic generation of plots comparing models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:

```bash
python simulate.py
```

This will:
1. Run both lumped and distributed parameter simulations
2. Generate comparison plots
3. Save results to a timestamped directory under `results/`

### Customizing Parameters

Edit the `DEFAULT_PARAMS` dictionary in `simulate.py` to modify:

- **Material Properties**: `cp_s`, `cp_l`, `k_s`, `k_l`, `rho`, `L_f`, `Tm`, `dT`
- **Geometry**: `length`, `n_nodes`, `A_surface`
- **Simulation**: `dt`, `total_days`
- **Boundary Conditions**: `T_init`, `T_hot`, `T_amb`, `h_conv`
- **Solar Radiation**: `use_solar`, `I_solar`, `alpha_abs`

### Programmatic Usage

```python
from simulate import simulate_lumped, simulate_distributed, plot_results

# Run with default parameters
results = simulate_lumped()

# Run with custom parameters
custom_params = {
    'T_hot': 70.0,
    'total_days': 5.0,
    'length': 0.1,
}
results = simulate_distributed(custom_params)

# Plot results
plot_results(results, save_dir='my_results')
```

## Output

The simulation produces:
- **Mass Fraction vs Time**: Shows the characteristic melting curve
- **Temperature vs Time**: Average PCM temperature
- **Rate of Melting**: Derivative of mass fraction (shows decreasing slope)
- **Temperature Profiles**: Spatial temperature distribution at different times (distributed model only)

## Physics

The solver implements the enthalpy method for the 1D Stefan problem:

$$\rho \frac{\partial H}{\partial t} = \frac{\partial}{\partial x}\left(k \frac{\partial T}{\partial x}\right)$$

Where:
- $H$ is specific enthalpy
- $T$ is temperature  
- $k$ is thermal conductivity
- $\rho$ is density

The enthalpy-temperature relationship includes:
- Sensible heat in solid phase: $H = \rho c_{p,s} (T - T_{ref})$
- Latent heat during phase change: $H = H_{solid} + f \cdot \rho L_f$
- Sensible heat in liquid phase: $H = H_{liquid} + \rho c_{p,l} (T - T_m)$

## License

See [LICENSE](LICENSE) for details.