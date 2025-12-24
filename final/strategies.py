from abc import ABC, abstractmethod
import numpy as np
from utils.computations import (
    get_gradient_func,
    get_symbolic_hessian,
)
from utils.armijo import armijo_rule
import sympy as sp


class OptimizationStrategy(ABC):
    """
    Clase base abstracta para las estrategias de optimización.
    """

    @abstractmethod
    def optimize(self, f, x0, t=1, max_iter=1000, epsilon=1e-6, **kwargs):
        """
        Ejecuta el algoritmo de optimización.

        Args:
            f: La función objetivo (callable o simbólica, según implementación).
            x0: Punto inicial (numpy array).
            t: Parámetro de paso inicial (float).
            max_iter: Número máximo de iteraciones (int).
            epsilon: Tolerancia para el criterio de parada (float).
            **kwargs: Argumentos adicionales específicos del algoritmo (tolerancia, max_iter, etc.).

        Returns:
            dict: Un diccionario con los resultados, por ejemplo:
                  {'x_opt': punto_optimo, 'f_opt': valor_optimo, 'path': historial_puntos}
        """
        pass


class UnconstrainedStrategy(OptimizationStrategy):
    def optimize(self, f, x_0, t=1, max_iter=1000, epsilon=1e-6, **kwargs):
        x, y = sp.symbols("x y")
        vars_list = [x, y]

        f_func = sp.lambdify(vars_list, f, "numpy")

        def f_wrapper(p):
            return f_func(p[0], p[1])

        grad_func = get_gradient_func(f, vars_list)

        def grad_wrapper(p):
            g = grad_func(p[0], p[1])
            return np.array(g, dtype=float).flatten()

        self._setup_specific(f, vars_list)

        x_k = np.array(x_0, dtype=float)
        path = [x_k.copy()]
        grad_f_val = grad_wrapper(x_k)

        self._init_state(x_k)

        for _ in range(max_iter):
            if np.linalg.norm(grad_f_val) < epsilon:
                break

            d_k = self._get_direction(x_k, grad_f_val)
            step_size = self._get_step_size(f_wrapper, grad_f_val, x_k, d_k, t)

            x_new = x_k + step_size * d_k
            grad_f_new = grad_wrapper(x_new)

            self._update_state(x_k, x_new, grad_f_val, grad_f_new)

            x_k = x_new
            grad_f_val = grad_f_new
            path.append(x_k.copy())

        return {
            "x_opt": x_k,
            "f_opt": f_wrapper(x_k),
            "path": np.array(path),
            "message": f"Optimización completada ({self.__class__.__name__})",
        }

    def _setup_specific(self, f, vars_list):
        pass

    def _init_state(self, x_k):
        pass

    @abstractmethod
    def _get_direction(self, x_k, grad_f_val):
        pass

    def _get_step_size(self, f, grad_f_val, x_k, d_k, t):
        return armijo_rule(f, grad_f_val, x_k, d_k, alpha=t)

    def _update_state(self, x_k, x_new, grad_f_val, grad_f_new):
        pass


# --- Optimización Irrestricta ---


class GradientDescentStrategy(UnconstrainedStrategy):
    def _get_direction(self, x_k, grad_f_val):
        return -grad_f_val


class NewtonStrategy(UnconstrainedStrategy):
    def _setup_specific(self, f, vars_list):
        hess_sym = get_symbolic_hessian(f, vars_list)
        hess_func = sp.lambdify(vars_list, hess_sym, "numpy")

        def hess_wrapper(p):
            h = hess_func(p[0], p[1])
            return np.array(h, dtype=float)

        self.hess_wrapper = hess_wrapper

    def _get_direction(self, x_k, grad_f_val):
        H_k = self.hess_wrapper(x_k)
        try:
            return np.linalg.solve(H_k, -grad_f_val)
        except np.linalg.LinAlgError: # Hessiano es singular
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
            self.B = (
                self.B
                + np.outer(y_k, y_k) / np.dot(y_k, s_k)
                - np.outer(self.B @ s_k, self.B @ s_k) / (s_k @ self.B @ s_k)
            )
            grad_f_val = grad_f_new


class NonlinearConjugateGradientStrategy(OptimizationStrategy):
    def optimize(self, f, x0, **kwargs):
        return {
            "x_opt": x0,
            "f_opt": 0.0,
            "path": np.array([x0]),
            "message": "Ejecución simulada de Gradientes Conjugados No Lineal",
        }


# --- Restricciones Fáciles ---


class ProjectedGradientStrategy(OptimizationStrategy):
    def optimize(self, f, x0, constraints=None, **kwargs):
        return {
            "x_opt": x0,
            "f_opt": 0.0,
            "path": np.array([x0]),
            "message": "Ejecución simulada de Gradiente Proyectado",
        }


# --- Restricciones Generales ---


class AugmentedLagrangianStrategy(OptimizationStrategy):
    def optimize(self, f, x0, constraints=None, **kwargs):
        return {
            "x_opt": x0,
            "f_opt": 0.0,
            "path": np.array([x0]),
            "message": "Ejecución simulada de Lagrangiano Aumentado",
        }


class SQPStrategy(OptimizationStrategy):
    def optimize(self, f, x0, constraints=None, **kwargs):
        return {
            "x_opt": x0,
            "f_opt": 0.0,
            "path": np.array([x0]),
            "message": "Ejecución simulada de SQP",
        }
