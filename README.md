# CalTech2025_RD-CRN_Project

This repository contains my work as a Summer 2025 Software Engineering Intern in the Erik Winfree Lab at Caltech. The goal of this project is to support the development of a differentiable simulation library for trainable reaction-diffusion systems and neural cellular automata (NCA), inspired by biological self-organization and development.

## 🔬 Project Overview

The core objective is to explore and implement models that bridge neural networks with chemical and spatial computation. This includes training neural networks in JAX, validating biological pattern formation via NCAs, and contributing to a broader framework for reaction-diffusion-based computation.

## 📂 Contents

### 1. `iris_nn/`  
A simple neural network implemented in JAX to classify the classic Iris dataset. Serves as an initial exploration of JAX’s functional style and autodiff features.

### 2. `nca_jax/`  
A JAX-based reimplementation of neural cellular automata (inspired by Mordvintsev’s work), including:
- Differentiable update rules
- Test cases for model behavior
- Training loop structure for future RD extensions

### 3. `tbd/`  
A third major component is currently under development — likely to include experiment modules or tutorial-style examples for a larger simulation framework.

## 🧪 Goals

- Validate and reproduce key results from the NCA literature  
- Contribute experiments, test cases, and examples to a reusable JAX/SymPy-based simulation library  
- Support accessible, reproducible research in programmable chemical systems  

## 📌 About

This project is conducted under the mentorship of researchers in the Erik Winfree Lab at Caltech, focusing on the intersection of ML, synthetic biology, and self-organizing systems.

---

> Feel free to reach out with questions or ideas!
