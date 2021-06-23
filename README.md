# Gradient-Field-Refinement-GAN
## Overview

This project uses Generative Adversarial Networks to resolve a low-resolution gradient field into a high-resolution gradient field. This has been **VERY** helpful in my research. In situations where it is desired to reconstruct a surface from a gradient field, this greatly helps with the accuracy of that reconstruction. Much better than just interpolating where the surface function is not necessarally smooth.

The GAN architecture used is based off of the [SRGAN paper](https://arxiv.org/abs/1609.04802).

## Getting Started

Once you have the code, set up a virtual environment if you would like and install the necessary libraries by running the command below.
```bat
pip install -r /path/to/requirements.txt
```
