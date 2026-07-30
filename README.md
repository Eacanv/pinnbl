# Physics-Informed Neural Networks for the Buckley–Leverett Equation

Reference implementation of three PINN formulations for the forward
Buckley–Leverett problem, accompanying the manuscript *"A Comparative Study of
Physics-Informed Neural Network Approximations for the Buckley–Leverett
Equation"* (JCMDS).

## Formulations

- **Standard PINN** — residual of the Buckley–Leverett equation, no additional
  regularization.
- **PINN+ε (artificial viscosity)** — adds a diffusive term ε·∂²S/∂x² to the
  residual to stabilize the front.
- **PINN+σ (entropy-constrained)** — enforces the Oleinik entropy condition
  through a corrected flux built from the Welge construction.

## Reference solution

The semi-analytical Buckley–Leverett solution (Welge construction) used as ground
truth for all error metrics is included.

## Requirements

- Python 3.10+
- TensorFlow 2.10
- NumPy, Matplotlib
