# Reproducibility in Machine Learning

This repository contains a mini-project focused on reproducing the results of the Deep Neural Decision Trees (DNDT) model, as outlined in the 2018 paper *Deep Neural Decision Trees*. The project evaluates the claims of interpretability, performance, and simplicity of DNDT compared to traditional Decision Trees (DT) and Neural Networks (NN).


## Overview
The goal of this project is to reproduce the DNDT model's results and evaluate its performance on multiple tabular datasets. The DNDT model combines the interpretability of Decision Trees with the power of Neural Networks, using gradient descent for training and GPU acceleration for scalability.


## Datasets
- **Iris Dataset**
- **Titanic Dataset**
- **Pima Indian Diabetes Dataset**
- **Breast Cancer Wisconsin Dataset**

Each dataset was preprocessed and analyzed to ensure compatibility with the DNDT model.

## Methodology
1. **Model Implementation**: DNDT, DT, and NN were implemented using PyTorch.
2. **Reproducibility Scope**: Focused on claims regarding DNDT’s performance and interpretability.
3. **Experimental Setup**: Compared normalized vs. unnormalized data, hyperparameter tuning, and GPU acceleration.



## Results
- DNDT showed competitive accuracy but was dataset-dependent.
- Interpretability through tree visualization was successfully demonstrated.
- Challenges arose from incomplete details in the original paper (e.g., missing hyperparameters).


## References
- [Deep Neural Decision Trees Paper](https://arxiv.org/pdf/1806.06988v1.pdf)
