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


def get_hessian(f_sym, vars_list):
    """
    Compute the Hessian of the function f_sym with respect to the given variables.

    Parameters:
    f_sym : sympy expression
        The symbolic representation of the objective function.
    vars_list : list of sympy symbols
        The variables with respect to which to compute the Hessian.

    Returns:
    callable
        A function that computes the Hessian matrix of f_sym.
    """
    n = len(vars_list)
    hessian = sp.Matrix(n, n, lambda i, j: sp.diff(f_sym, vars_list[i], vars_list[j]))
    return sp.lambdify(vars_list, hessian, "numpy")

def box_projection(x, bounds):
    """
    Project the point x onto the box defined by bounds.

    Parameters:
    x : np.ndarray
        The point to be projected.
    bounds : list of tuples
        The bounds for each dimension, where each tuple is (lower_bound, upper_bound).

    Returns:
    np.ndarray
        The projected point.
    """
    projected_x = np.copy(x)
    for i in range(len(x)):
        lower_bound, upper_bound = bounds[i]
        projected_x[i] = np.clip(projected_x[i], lower_bound, upper_bound)
    return projected_x