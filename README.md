# 🌍 Pollution Dispersion Modeling using Advection-Dispersion Equation

This repository presents a study and implementation of **1D numerical schemes** for modeling pollutant transport using the **Advection-Diffusion Equation (ADE)**. The project applies **Physics-Informed Neural Networks (PINNs)** to solve both forward and inverse problems, with a focus on identifying pollutant sources from limited data.

## 📘 Overview

Pollution transport in environmental systems is governed by two key processes:
- **Advection** — transport due to the bulk motion of fluid.
- **Diffusion** — spreading caused by concentration gradients.

This work formulates and solves the ADE in 1D to simulate pollution spread and to identify unknown source locations using neural networks that are informed by the physics of the problem.

## 📌 Problem Statement

Given sparse or noisy observational data, can we:
- Accurately predict pollutant concentration across time and space?
- Identify the source of contamination using inverse modeling techniques?

We address this using the Advection-Diffusion PDE:

\[
\frac{\partial C}{\partial t} + u \frac{\partial C}{\partial x} = D \frac{\partial^2 C}{\partial x^2} - kC
\]

Where:
- \( C(x,t) \): pollutant concentration
- \( u \): advection velocity
- \( D \): diffusion coefficient
- \( k \): decay rate

## 🌱 Environmental Motivation

ADE-based modeling plays a vital role in:
- 🛢 Oil spill tracking
- 🌬️ Air quality analysis
- 💨 CO₂ leak detection (CCUS technologies)

Understanding source identification is critical in **environmental forensics**, especially when data is limited.

## 🔬 Mathematical Background

- Derived from conservation laws and Fick’s diffusion law.
- Handles **inhomogeneous source terms**, approximated using a Gaussian function.
- Boundary conditions: **Homogeneous Dirichlet (absorbing walls)** used in this study.

## 🤖 PINN Approach

We solve the ADE using **Physics-Informed Neural Networks**, which:
- Do not require mesh generation
- Encode the PDE and boundary/initial conditions into the loss function
- Are robust against sparse and noisy data

### Loss Function Components
- **PDE Residual** \( R(x,t) \)
- **Initial Condition Loss**
- **Boundary Condition Loss**

The model is trained to minimize the total loss:

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{BC}} + \mathcal{L}_{\text{IC}}
\]

## 📈 Results

- Forward simulation: pollutant spreads downstream due to advection and diffuses out.
- Inverse modeling: PINN accurately identifies source characteristics.
- Comparison GIFs visualize the contrast between pure advection and full ADE dynamics.

## 📂 File Structure

```
.
├── data/               # Synthetic or observational data (if any)
├── src/                # PINN implementation and solvers
├── plots/              # Figures and GIFs from simulations
├── README.md           # Project overview
└── report.pdf          # Detailed project documentation
```

## 📚 References

- Cantor, B. (2020). *The Equations of Materials*.
- Ewing, R., Wang, H. (2001). Numerical methods for advection-dominated PDEs.
- Gustafsson, B., et al. (2013). *Time Dependent Problems and Difference Methods*.

---

## 👩‍💻 Authors

- **Prakhar (B23091)**
- **Nishita (B23403)**  
Indian Institute of Technology Mandi, May 2025
