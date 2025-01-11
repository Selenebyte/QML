# Analyzing the Impact of Parameter Variations and Noise on Fourier Coefficients in Variational Quantum Machine Learning Models

<p align="center">
<a href="https://github.com/Selenebyte/QML/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This GitHub page contains the code used in "Analyzing the Impact of Parameter Variations and Noise on Fourier Coefficients in Variational Quantum Machine Learning Models".

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## About

This project analyzes how parameter variations and noise impact a variational quantum machine learning model. It uses correlations between layers of the quantum models to show that they have an inverse relationship with the model's performance. The correlations used are Pearson, Distance, Cross, and Mutual information. The paper also describes how higher frequencies are inaccessible during training for an x-axis Pauli rotation and how high amounts of noise contribute to all frequencies' inaccessibility. Whereas low amounts of noise work as regularization during training.

## Installation

1. **Install Dependencies:**

    Python 3.12.8.

    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Project:**

    Open the Jupyter Notebook: main.ipynb

## Usage

"model_framework.py" allows for the creation of other models using different trainable circuit blocks and data encoding circuit blocks.

Datasets and parameters are configured in the Python notebook, and models can subsequently be trained. Various visualization tools are provided.

## License

Distributed under the [MIT](LICENSE).