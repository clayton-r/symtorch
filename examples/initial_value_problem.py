from sympy import symbols, Matrix
import torch
import numpy as np
from symtorch.integrate import IVP

x, y, a, b, c, d = symbols('x y a b c d')
X = x, y
args = a, b, c, d

lotkavolterra = Matrix([[a*x - b*x*y], [c*x*y - d*y]])
y0 = torch.tensor([10, 5]).float().view(-1, 1)

model = IVP(func=lotkavolterra, x=X, args_symbols=args, args_values=torch.tensor([1.5, 1, 3, 1]).float())
solution = model(y0=y0, t_span=torch.linspace(0, 15, 300))

print(solution.squeeze(2))


