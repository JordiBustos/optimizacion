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
    def optimize(self, f, x0, **kwargs):
        """
        Ejecuta el algoritmo de optimización.

        Args:
            f: La función objetivo (callable o simbólica, según implementación).
            x0: Punto inicial (numpy array).
            **kwargs: Argumentos adicionales específicos del algoritmo (tolerancia, max_iter, etc.).

        Returns:
            dict: Un diccionario con los resultados, por ejemplo:
                  {'x_opt': punto_optimo, 'f_opt': valor_optimo, 'path': historial_puntos}
        """
        pass


# --- Optimización Irrestricta ---


class GradientDescentStrategy(OptimizationStrategy):
    def optimize(self, f, x_0, t=1, max_iter=1000, epsilon=1e-6, **kwargs):
        x, y = sp.symbols("x y")
        vars_list = [x, y]

        f_func = sp.lambdify(vars_list, f, "numpy")

        def f_wrapper(p):
            return f_func(p[0], p[1])

        grad_func = get_gradient_func(f, vars_list)

        def grad_wrapper(p):
            g = grad_func(p[0], p[1])
            return np.array(g, dtype=float)

        x_k = np.array(x_0, dtype=float)
        path = [x_k.copy()]

        grad_f_val = grad_wrapper(x_k)

        for _ in range(max_iter):
            if np.linalg.norm(grad_f_val) < epsilon:
                break

            d_k = -grad_f_val

            step_size = armijo_rule(f_wrapper, grad_f_val, x_k, d_k, alpha=t)

            x_k = x_k + step_size * d_k
            grad_f_val = grad_wrapper(x_k)
            path.append(x_k.copy())

        return {
            "x_opt": x_k,
            "f_opt": f_wrapper(x_k),
            "path": np.array(path),
            "message": "Optimización completada",
        }


class NewtonStrategy(OptimizationStrategy):
    def optimize(self, f, x_0, t=1, max_iter=1000, epsilon=1e-6, **kwargs):
        x, y = sp.symbols("x y")
        vars_list = [x, y]

        f_func = sp.lambdify(vars_list, f, "numpy")

        def f_wrapper(p):
            return f_func(p[0], p[1])

        grad_func = get_gradient_func(f, vars_list)
        hess_sym = get_symbolic_hessian(f, vars_list)
        hess_func = sp.lambdify(vars_list, hess_sym, "numpy")

        def grad_wrapper(p):
            g = grad_func(p[0], p[1])
            return np.array(g, dtype=float).flatten()

        def hess_wrapper(p):
            h = hess_func(p[0], p[1])
            return np.array(h, dtype=float)

        x_k = np.array(x_0, dtype=float)
        path = [x_k.copy()]

        grad_f_val = grad_wrapper(x_k)

        for _ in range(max_iter):
            if np.linalg.norm(grad_f_val) < epsilon:
                break

            H_k = hess_wrapper(x_k)

            try:
                d_k = np.linalg.solve(H_k, -grad_f_val)
            except np.linalg.LinAlgError:
                d_k = -grad_f_val

            step_size = armijo_rule(f_wrapper, grad_f_val, x_k, d_k, alpha=1.0)

            x_k = x_k + step_size * d_k
            grad_f_val = grad_wrapper(x_k)
            path.append(x_k.copy())

        return {
            "x_opt": x_k,
            "f_opt": f_wrapper(x_k),
            "path": np.array(path),
            "message": "Optimización completada (Newton)",
        }


class QuasiNewtonArmijoStrategy(OptimizationStrategy):
    def optimize(self, f, x0, **kwargs):
        return {
            "x_opt": x0,
            "f_opt": 0.0,
            "path": np.array([x0]),
            "message": "Ejecución simulada de Quasi-Newton (Armijo)",
        }


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
