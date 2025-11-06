# Solving the Damped Harmonic Oscillator using Physics-Informed Neural Networks

## Overview
This project implements and evaluates a **Physics-Informed Neural Network (PINN)** to solve the second-order ordinary differential equation (ODE) governing a damped harmonic oscillator. The network, a **Multi-Layer Perceptron (MLP)**, is trained using a hybrid loss function that minimizes both the error from known initial conditions and the physics residual from the ODE, which is evaluated at unsupervised collocation points.

A key contribution of this project is the implementation of a **linear annealing strategy** for the physics loss weight. This technique is introduced to address training imbalances between loss components. The model's performance is validated against the known analytical solution and a standard numerical solver (`SciPy`), demonstrating that the annealed PINN achieves significantly higher accuracy than both the baseline PINN and the numerical method.

## Key Features

* **PINN Implementation:** A complete implementation of a PINN in PyTorch to solve an ODE.
* **Annealing Loss Strategy:** Implements a linear annealing weight for the physics loss component to improve training stability and final accuracy.
* **Comparative Analysis:** Provides a quantitative and visual comparison between:
    * The standard PINN 
    * The improved PINN with annealing
    * A traditional numerical solver 
    * The exact analytical solution

## Methodology

1.  **Network Architecture:** A Multi-Layer Perceptron (MLP) with 4 hidden layers and 32 neurons each, using the `tanh` activation function for differentiability.
2.  **Hybrid Loss Function:** The loss is a combination of two MSE components:
    * **Initial Condition Loss ($\mathcal{L}_{ic}$):** Enforces the known starting conditions $x(0)=1.0$ and $\dot{x}(0)=0.0$.
    * **Physics Loss ($\mathcal{L}_{phys}$):** Enforces the ODE's residual equation (Eq. 3) at 100 random collocation points using automatic differentiation.
3.  **Annealing:** The total loss $\mathcal{L}_{total} = w_{ic}\mathcal{L}_{ic} + w_{phys}\mathcal{L}_{phys}$ is used, where $w_{phys}$ is linearly increased from 0.01 to 1.0 over the first 5000 epochs to stabilize training.

## Results

The project successfully demonstrates that the PINN with annealing not only outperforms the baseline PINN but also achieves higher accuracy than the standard SciPy numerical solver for this problem.

### Quantitative Comparison

As shown in the Table , the annealed PINN achieved the lowest error:

| Model / Method | MSE | RMSE |
| :--- | :--- | :--- |
| PINN (standard) | $3.4451 \times 10^{-7}$ | $5.8695 \times 10^{-4}$ |
| **PINN (Annealed)** | **$9.4700 \times 10^{-9}$** | **$9.7295 \times 10^{-5}$** |
| SciPy Solver | $1.9418 \times 10^{-7}$ | $4.4066 \times 10^{-4}$ |

### Visual Comparison

The final plot shows a near-perfect overlap between the annealed PINN prediction and the ground truth analytical solution, visually confirming its high accuracy.

## Running the Code

The entire implementation, training, and analysis is contained in the Jupyter Notebook: `proj_notebook.ipynb`.

1.  Open `proj_notebook.ipynb`.
2.  Run all cells sequentially to train the models, generate the results.

