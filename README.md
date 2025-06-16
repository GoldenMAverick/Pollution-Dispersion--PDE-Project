# Pollution-Dispersion--PDE-Project\This repository presents a study and implementation of 1D numerical schemes for modeling pollutant transport using the Advection-Diffusion Equation (ADE). The project applies Physics-Informed Neural Networks (PINNs) to solve both forward and inverse problems, with a focus on identifying pollutant sources from limited data.

📘 Overview
Pollution transport in environmental systems is governed by two key processes:

Advection — transport due to the bulk motion of fluid.

Diffusion — spreading caused by concentration gradients.

This work formulates and solves the ADE in 1D to simulate pollution spread and to identify unknown source locations using neural networks that are informed by the physics of the problem.

📌 Problem Statement
Given sparse or noisy observational data, can we:

Accurately predict pollutant concentration across time and space?

Identify the source of contamination using inverse modeling techniques?

We address this using the Advection-Diffusion PDE:

∂
𝐶
∂
𝑡
+
𝑢
∂
𝐶
∂
𝑥
=
𝐷
∂
2
𝐶
∂
𝑥
2
−
𝑘
𝐶
∂t
∂C
​
 +u 
∂x
∂C
​
 =D 
∂x 
2
 
∂ 
2
 C
​
 −kC
Where:

𝐶
(
𝑥
,
𝑡
)
C(x,t): pollutant concentration

𝑢
u: advection velocity

𝐷
D: diffusion coefficient

𝑘
k: decay rate

🌱 Environmental Motivation
ADE-based modeling plays a vital role in:

🛢 Oil spill tracking

🌬️ Air quality analysis

💨 CO₂ leak detection (CCUS technologies)

Understanding source identification is critical in environmental forensics, especially when data is limited.

🔬 Mathematical Background
Derived from conservation laws and Fick’s diffusion law.

Handles inhomogeneous source terms, approximated using a Gaussian function.

Boundary conditions: Homogeneous Dirichlet (absorbing walls) used in this study.

🤖 PINN Approach
We solve the ADE using Physics-Informed Neural Networks, which:

Do not require mesh generation

Encode the PDE and boundary/initial conditions into the loss function

Are robust against sparse and noisy data

Loss Function Components
PDE Residual 
𝑅
(
𝑥
,
𝑡
)
R(x,t)

Initial Condition Loss

Boundary Condition Loss

The model is trained to minimize the total loss:

𝐿
total
=
𝐿
PDE
+
𝐿
BC
+
𝐿
IC
L 
total
​
 =L 
PDE
​
 +L 
BC
​
 +L 
IC
​
 
📈 Results
Forward simulation: pollutant spreads downstream due to advection and diffuses out.

Inverse modeling: PINN accurately identifies source characteristics.

Comparison GIFs visualize the contrast between pure advection and full ADE dynamics.
