from sympy import symbols, Matrix
import torch
import numpy as np
from scipy.integrate import solve_ivp
from symtorch.integrate import IVP


def scipy_lotkavolterra(t, z, a, b, c, d):
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]


def symtorch_lotkavolterra():
    x, y, a, b, c, d = symbols('x y a b c d')
    X = x, y
    args = a, b, c, d
    return Matrix([[a*x - b*x*y], [-c*y + d*x*y]]), X, args


def test_ivp():
    sym_lotkavolterra, X, args = symtorch_lotkavolterra()

    y0 = torch.tensor([10, 5]).double().unsqueeze(1)

    model = IVP(func=sym_lotkavolterra, x=X, method='dopri5',
                args_symbols=args, args_values=torch.tensor([1.5, 1, 3, 1]).double())

    sym_z = model(y0=y0, t_span=torch.linspace(0, 15, 300)).squeeze(2).T

    sol = solve_ivp(scipy_lotkavolterra, [0, 15], [10, 5], args=(1.5, 1, 3, 1), dense_output=True, method='RK45')
    t = np.linspace(0, 15, 300)
    sci_z = torch.tensor(sol.sol(t))

    assert torch.median(abs(sym_z - sci_z)) < 0.05
