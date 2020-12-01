# symtorch
A PyTorch and SymPy based library for the use of symbolic neural differential equations. 

<p align="center">
<img src="media/Logo.png" width="150" height="150">
</p>
<div align="center">
</div>

While neural network-based function approximation is incredible for its ability to approximate any function, where we know the exact function, symtorch can be useful.

### Feature roadmap

* **CUDA:** This implementation currently only runs on CPU ⬜️
* **Stochastic Neural ODE solver:** torchSDE ⬜️
* **neural CDE** ⬜️
* **Stable API**  ⬜️
* **Incorporation into torchdyn:** so that pytorch differential equation work can live in one place  ⬜️

### Example
Please see the examples directory for how to use this within the context of a pytorch Model

```python:
from sympy import symbols, lambdify, Matrix

import torch
import inspect
from matplotlib import pyplot as plt
from torchdiffeq import odeint


K_ = symbols('k1 k2 k3 k4')
S_ = symbols('S1 S2 S3 S4')
t_ = symbols('t')
k1, k2, k3, k4 = K_
S1, S2, S3, S4 = S_
sys = Matrix([[-S1*k1], [S1*k1 - S2*k2 + S3*k3], [S2*k2 - S3*k3 - S3*k4], [S3*k4]])

array2mat = [{'ImmutableDenseMatrix': torch.tensor}, 'torch']
f = lambdify([t_, S_, K_], sys, modules=array2mat)


def lambda_f(t, s):
    k = torch.tensor([0.1, 0.5, 0.5, 0.5])
    return f(t, s, k)


init = torch.tensor([100.0, 0.0, 0.0, 0.0]).view(-1, 1)
t = torch.linspace(0, 50, 101)
sola = odeint(func=lambda_f, y0=init, t=t, method='rk4')
```
