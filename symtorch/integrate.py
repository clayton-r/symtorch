import torch
import torch.nn as nn
import sympy
import inspect

import torchdiffeq
from symtorch.symbolic import Symbolic


class IVP(nn.Module):
    def __init__(self, func: sympy.Matrix, x: sympy.symbols, method: str = 'rk4',
                 args_symbols: sympy.symbols = None, args_values: torch.tensor = None):
        super(IVP, self).__init__()

        t = sympy.symbols('t')
        self.method = method
        self.symbolic_function = Symbolic(func=func, x=x, t=t, args_symbols=args_symbols, args_init_values=args_values)

        if method == 'rk4' or 'dopri5':
            self.odeint = torchdiffeq.odeint

    def forward(self, y0, t_span):
        return self.odeint(func=self.symbolic_function, y0=y0, t=t_span, method=self.method)





