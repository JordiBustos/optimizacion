from numpy import dot


# Naive implementation of https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html
# Based on Nocedal and Wright, 3.5 & 3.6.
def line_search(
    f,
    grad_f_val,
    xk,
    dk,
    alpha=1.0,
    sigma=0.25,
    sigma_2=0.9,
    grad_wrapper=None,
):
    """
    Encuentra un tamaño de paso que cumple la condición de Armijo y Wolfe fuerte.

    Parameters:
    - f: Function to minimize.
    - grad_f_val: Gradient of f at xk.
    - xk: Current point.
    - dk: Search direction.
    - alpha: Initial step size.
    - sigma: Parameter for Armijo condition.
    - sigma_2: Parameter for curvature condition in Strong Wolfe.
    - grad_wrapper: Function to compute gradient at a point.
    """
    phi_0 = f(xk)
    g_0 = dot(grad_f_val, dk)

    # Si la dirección no es descendente, invertirla
    if g_0 >= 0:
        dk = -dk
        g_0 = -g_0

    alpha_prev = 0
    phi_prev = phi_0

    for i in range(100):
        x_new = xk + alpha * dk
        phi_curr = f(x_new)

        # Armijo
        if (phi_curr > phi_0 + sigma * alpha * g_0) or (i > 0 and phi_curr >= phi_prev):
            return _zoom(
                alpha_prev, alpha, f, grad_wrapper, xk, dk, phi_0, g_0, sigma, sigma_2
            )

        g_curr = dot(grad_wrapper(x_new), dk)

        if abs(g_curr) <= -sigma_2 * g_0:
            return alpha

        # Chequear si no nos pasamos con un step grande
        if g_curr >= 0:
            return _zoom(
                alpha, alpha_prev, f, grad_wrapper, xk, dk, phi_0, g_0, sigma, sigma_2
            )

        alpha_prev = alpha
        phi_prev = phi_curr
        alpha = alpha * 2.0

    return alpha


def _zoom(a_lo, a_hi, f, grad_f, xk, dk, phi_0, g_0, c1, c2):
    """
    Zoom function to find alpha within the interval [a_lo, a_hi].
    """
    for _ in range(20):
        alpha_j = (a_lo + a_hi) / 2.0

        x_new = xk + alpha_j * dk
        phi_curr = f(x_new)

        x_lo = xk + a_lo * dk
        phi_lo = f(x_lo)

        if (phi_curr > phi_0 + c1 * alpha_j * g_0) or (phi_curr >= phi_lo):
            a_hi = alpha_j
        else:
            g_curr = dot(grad_f(x_new), dk)

            if abs(g_curr) <= -c2 * g_0:
                return alpha_j

            if g_curr * (a_hi - a_lo) >= 0:
                a_hi = a_lo

            a_lo = alpha_j

    return a_lo
