# Multistage PINNs for Kovasznay Flow

CEE 292 Scientific Machine Learning project implementing physics-informed neural networks in JAX for steady, incompressible Navier–Stokes flow.

## Overview

This project implements the Kovasznay-flow benchmark from NSFnets and investigates:

- Adam, Adam + L-BFGS-B, and SOAP optimization.
- Constant, plateau, cosine, and piecewise learning-rate schedules.
- A two-stage PINN that learns a correction to a frozen first-stage solution.

The benchmark uses Reynolds number 40 and predicts velocity components `u`, `v`, and pressure `p`.

## Run in Google Colab

1. Open the project notebook in Colab.
2. Select a GPU runtime.
3. Run the installation cell before importing packages.
4. Execute the setup and experiment sections in order.
5. Set the output directory to your preferred location.

Full experiments may require several minutes per run.

## Data and results

The notebook generates interior collocation points and analytical boundary velocities. No labeled interior solution data are used for training.

Saved outputs include training points, model parameters, loss histories, solution errors, PDE residuals, and figures. Pressure is mean-aligned before evaluation.

## Main result

In the recorded experiment, two-stage SOAP reduced the geometric mean of relative solution errors by approximately **10.3×** compared with single-stage SOAP at approximately equal training time.

The two-stage method also uses more parameters and denser correction-stage training data; this is a comparison of complete methods.

## References

- [Jin et al. (2021), NSFnets](https://arxiv.org/abs/2003.06496)
- [Wang and Lai (2024), Multi-stage neural networks](https://arxiv.org/abs/2307.08934)
- [Vyas et al., SOAP](https://arxiv.org/abs/2409.11321)
- [SOAP JAX implementation](https://github.com/haydn-jones/SOAP_JAX)
