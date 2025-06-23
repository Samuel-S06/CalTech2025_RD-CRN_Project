# Neural Network Classifier for Iris Dataset using JAX

This project implements a **neural network classifier** for the classic **Iris dataset** using **JAX** and **Optax** for accelerated numerical computation and optimization. It demonstrates how to build, train, and evaluate a basic multi-layer neural network from scratch in JAX.

## Features
- Neural network with **2 hidden layers** and **ReLU** activations
- Loss function with **cross-entropy** and **L2 regularization**
- **Adam optimizer** (via Optax)
- Visualization of **training loss** and **test accuracy** over epochs
- Runs fully on CPU or GPU depending on your JAX installation

## Requirements
- Python 3.10+
- JAX
- Optax
- Scikit-learn
- Matplotlib
- NumPy (comes with JAX)
  
You can install the required libraries using:

```bash
pip install --upgrade jax jaxlib optax scikit-learn matplotlib

