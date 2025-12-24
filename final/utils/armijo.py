import numpy as np

def armijo_rule(f, grad_f_val, xk, dk, alpha=1.0, beta=0.5, sigma=0.25):
    """
    Perform the Armijo rule (backtracking line search) to find an appropriate step size.

    Parameters:
    f : callable
        The objective function.
    grad_f_val : ndarray
        Gradient of the function at point xk.
    xk : ndarray
        Current point.
    dk : ndarray
        Search direction.
    alpha : float, optional
        Initial step size (default is 1.0).
    beta : float, optional
        Step size reduction factor (default is 0.5).
    sigma : float, optional
        Sufficient decrease parameter (default is 1e-4).

    Returns:
    float
        The step size that satisfies the Armijo condition.
    """
    fk = f(xk)
    
    while f(xk + alpha * dk) > fk + sigma * alpha * np.dot(grad_f_val, dk):
        alpha *= beta
    
    return alpha
