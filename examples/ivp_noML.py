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
np_sola = sola.view(101, 4).numpy()

fig0, ax0 = plt.subplots(figsize=(8, 3))
ax0.plot(t, np_sola)
ax0.set_title('Numerical Solution Torch')
plt.savefig(fname='plot_torch.png')
