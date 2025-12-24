import numpy as np
import sympy as sp


def get_gradient_func(f_sym, vars):
    """
    Compute the symbolic gradient of the function f_sym with respect to the given variables.

    Parameters:
    f_sym : sympy expression
        The symbolic representation of the objective function.
    vars : list of sympy symbols
        The variables with respect to which to compute the gradient.

    Returns:
    list of sympy expressions
        The symbolic gradient of f_sym.
    """
    import sympy as sp

    grad = [sp.diff(f_sym, var) for var in vars]
    return sp.lambdify(vars, grad, "numpy")


def get_numerical_hessian(f, x, h=1e-5):
    """
    Compute the numerical Hessian of the function f at point x using finite differences.
    Parameters:
    f : callable
        The objective function.
    x : ndarray
        Point at which to compute the Hessian.
    h : float, optional
        Step size for finite difference (default is 1e-5).
    Returns:
    ndarray
        The numerical Hessian of f at x.
    """
    n = len(x)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_ijp = np.array(x, dtype=float)
            x_ijm = np.array(x, dtype=float)
            x_ipj = np.array(x, dtype=float)
            x_imj = np.array(x, dtype=float)

            x_ijp[i] += h
            x_ijp[j] += h

            x_ijm[i] += h
            x_ijm[j] -= h

            x_ipj[i] -= h
            x_ipj[j] += h

            x_imj[i] -= h
            x_imj[j] -= h

            hessian[i, j] = (f(x_ijp) - f(x_ijm) - f(x_ipj) + f(x_imj)) / (4 * h * h)
    return hessian


def get_symbolic_hessian(f_sym, vars):
    """
    Compute the symbolic Hessian of the function f_sym with respect to the given variables.

    Parameters:
    f_sym : sympy expression
        The symbolic representation of the objective function.
    vars : list of sympy symbols
        The variables with respect to which to compute the Hessian.

    Returns:
    sympy Matrix
        The symbolic Hessian of f_sym.
    """
    n = len(vars)
    hessian = sp.Matrix(n, n, lambda i, j: sp.diff(f_sym, vars[i], vars[j]))
    return hessian
