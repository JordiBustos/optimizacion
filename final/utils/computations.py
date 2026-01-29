from numpy import copy, clip, array
from sympy import diff, Matrix, lambdify


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
    grad = [diff(f_sym, var) for var in vars]
    return lambdify(vars, grad, "numpy")


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
    hessian = Matrix(n, n, lambda i, j: diff(f_sym, vars_list[i], vars_list[j]))
    return lambdify(vars_list, hessian, "numpy")


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
    projected_x = copy(x)
    for i in range(len(x)):
        lower_bound, upper_bound = bounds[i]
        projected_x[i] = clip(projected_x[i], lower_bound, upper_bound)
    return projected_x

def is_f_constant(f, x_0):
    """
    Check if the function f is constant around the point x_0.

    Parameters:
    f : callable
        The objective function.
    x_0 : np.ndarray
        The initial point of the algorithm. Just used to build the response.
    Returns:
    dict
        A dictionary indicating if the function is constant and a message.
    """
    if isinstance(f, (int, float)) or (
            hasattr(f, "free_symbols") and not f.free_symbols
        ):
            x_k = array(x_0, dtype=float)
            return {
                "is_constant": True,
                "x_opt": x_k,
                "f_opt": float(f),
                "path": array([x_k]),
                "message": "La función es constante",
            }

    return {"is_constant": False}