# LAVA: Data Valuation without Pre-Specified Learning Algorithms
![Python 3.8.10](https://img.shields.io/badge/python-3.8.10-DodgerBlue.svg?style=plastic)

This repository is the official implementation of the "[LAVA: Data Valuation without Pre-Specified Learning Algorithms](https://openreview.net/forum?id=JJuP86nBl4q)" (ICLR 2023). We propose LAVA: a novel model-agnostic framework to data valuation using a non-conventional, class-wise Wasserstein discrepancy. We further introduce an efficient way to measure datapoint contribution at no cost from the optimization solution.

## Getting Started

```python
import lava
```
Coming Soon.

## Examples

For better understanding of applying LAVA to data valuation, we have provided examples on [CIFAR-10](example-cifar10.ipynb) and [STL-10](example-stl10.ipynb).

## Checkpoints

The pretrained embedders are included in the folder ['checkpoint'](checkpoint).


## Optimal Transport Solver
 
This repo relies on the [OTDD](https://github.com/microsoft/otdd) implementation to compute the class-wise Wasserstein distance. </br>
We are immensely grateful to the authors of that project.

