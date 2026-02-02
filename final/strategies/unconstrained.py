from abc import abstractmethod
import numpy as np
from utils.computations import (
    get_gradient_func,
    get_hessian,
    is_f_constant,
)
from utils.armijo import armijo_rule
from utils.utils import build_algorithm_response
import sympy as sp
from .optimization_strategy import OptimizationStrategy

# Common symbols used across all strategies
_x, _y = sp.symbols("x y")
VARS_LIST = [_x, _y]


def _make_f_wrapper(f_func):
    """Create a wrapper function for f that takes a point array."""
    return lambda p: f_func(p[0], p[1])


def _make_grad_wrapper(grad_func):
    """Create a wrapper function for gradient that takes a point array."""
    def wrapper(p):
        g = grad_func(p[0], p[1])
        return np.array(g, dtype=float).flatten()
    return wrapper


class UnconstrainedStrategy(OptimizationStrategy):
    def optimize(
        self, f, x_0, t=1, max_iter=1000, epsilon=1e-6, beta=0.5, sigma=0.25, **kwargs
    ):
        is_constant = is_f_constant(f, x_0)
        if is_constant["is_constant"]:
            return is_constant

        f_func = sp.lambdify(VARS_LIST, f, "numpy")
        f_wrapper = _make_f_wrapper(f_func)

        grad_func = get_gradient_func(f, VARS_LIST)
        grad_wrapper = _make_grad_wrapper(grad_func)

        self._setup_specific(f, VARS_LIST)

        x_k = np.array(x_0, dtype=float)
        path = [x_k.copy()]
        grad_f_val = grad_wrapper(x_k)

        self._init_state(x_k)

        for i in range(max_iter):
            if np.linalg.norm(grad_f_val) < epsilon:
                break

            d_k = self._get_direction(x_k, grad_f_val)
            step_size = self._get_step_size(
                f_wrapper, grad_f_val, x_k, d_k, t, beta=beta, sigma=sigma, **kwargs
            )

            x_new = x_k + step_size * d_k
            grad_f_new = grad_wrapper(x_new)

            self._update_state(x_k, x_new, grad_f_val, grad_f_new)

            x_k = x_new
            grad_f_val = grad_f_new
            path.append(x_k.copy())

        return build_algorithm_response(
            x_k, f_wrapper, path, self.__class__.__name__, i
        )

    def _setup_specific(self, f, vars_list):
        pass

    def _init_state(self, x_k):
        pass

    @abstractmethod
    def _get_direction(self, x_k, grad_f_val):
        pass

    def _get_step_size(
        self, f, grad_f_val, x_k, d_k, t, beta=0.5, sigma=0.25, **kwargs
    ):
        return armijo_rule(f, grad_f_val, x_k, d_k, alpha=t, beta=beta, sigma=sigma)

    def _update_state(self, x_k, x_new, grad_f_val, grad_f_new):
        pass


# --- Optimización Irrestricta ---


class GradientDescentStrategy(UnconstrainedStrategy):
    def _get_direction(self, x_k, grad_f_val):
        return -grad_f_val


class NewtonStrategy(UnconstrainedStrategy):
    def _setup_specific(self, f, vars_list):
        hess_func = get_hessian(f, vars_list)
        self.hess_wrapper = lambda p: np.array(hess_func(p[0], p[1]), dtype=float)

    def _get_direction(self, x_k, grad_f_val):
        H_k = self.hess_wrapper(x_k)
        try:
            return np.linalg.solve(H_k, -grad_f_val)
        except np.linalg.LinAlgError:  # Hessiano es singular
            return -grad_f_val


class QuasiNewtonArmijoStrategy(UnconstrainedStrategy):
    def _init_state(self, x_k):
        self.B = np.eye(len(x_k))

    def _get_direction(self, x_k, grad_f_val):
        try:
            return -np.linalg.solve(self.B, grad_f_val)
        except np.linalg.LinAlgError:
            return -grad_f_val

    def _update_state(self, x_k, x_new, grad_f_val, grad_f_new):
        s_k = x_new - x_k
        y_k = grad_f_new - grad_f_val

        if np.dot(y_k, s_k) > 1e-10:
            # Fórmula de BFGS
            self.B = (
                self.B
                + np.outer(y_k, y_k) / np.dot(y_k, s_k)
                - np.outer(self.B @ s_k, self.B @ s_k) / (s_k @ self.B @ s_k)
            )
            grad_f_val = grad_f_new


class NonlinearConjugateGradientStrategy(UnconstrainedStrategy):
    def _init_state(self, x_k):
        self.prev_grad = None
        self.prev_direction = None

    def _get_direction(self, x_k, grad_f_val):
        if self.prev_grad is None:
            direction = -grad_f_val
        else:
            # Fórmula de Fletcher-Reeves
            beta_k = np.dot(grad_f_val, grad_f_val) / np.dot(
                self.prev_grad, self.prev_grad
            )
            direction = -grad_f_val + beta_k * self.prev_direction

        self.prev_grad = grad_f_val
        self.prev_direction = direction
        return direction
