import torch.nn as nn
import torch
import sympy
import inspect


class Symbolic(nn.Module):
    def __init__(self, func: sympy.Matrix, x: sympy.symbols, t: sympy.symbols,
                 args_symbols: sympy.symbols = None, args_init_values: torch.tensor = None):
        """
        Symbolic Expression Layer
        :param func: sympy Matrix of symbolic expressions with the shape of the input tensor
        :param x: sympy symbols in the shape of the input tensor
        :param x: time symbol
        :param args_symbols: additional variables to be processed as Parameters
        :param args_init_values: initialization of the additional variables
        """
        super(Symbolic, self).__init__()

        if args_symbols is not None:
            assert args_init_values is not None
        elif args_init_values is not None:
            assert args_symbols is not None

        dummy = sympy.symbols('d')
        array2mat = [{'ImmutableDenseMatrix': torch.tensor}, 'torch']

        if args_symbols is not None:
            self.f = sympy.lambdify([t, x, args_symbols], func, modules=array2mat)
        else:
            self.f = sympy.lambdify([t, x, dummy], func, modules=array2mat)

        self.k = nn.Parameter(args_init_values, requires_grad=True)

    def forward(self, t, y):
        return self.f(t, y, self.k)
