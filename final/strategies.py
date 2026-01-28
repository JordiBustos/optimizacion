from abc import ABC, abstractmethod
import numpy as np
from utils.computations import (
    get_gradient_func,
    get_hessian,
    box_projection,
)
from utils.armijo import armijo_rule
import sympy as sp


class OptimizationStrategy(ABC):
    """
    Clase base abstracta para las estrategias de optimización.
    """

    @abstractmethod
    def optimize(
        self, f, x_0, t=1, max_iter=1000, epsilon=1e-6, beta=0.5, sigma=0.25, **kwargs
    ):
        """
        Ejecuta el algoritmo de optimización.

        Args:
            f: La función objetivo (callable o simbólica, según implementación).
            x_0: Punto inicial (numpy array).
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
    def optimize(
        self, f, x_0, t=1, max_iter=1000, epsilon=1e-6, beta=0.5, sigma=0.25, **kwargs
    ):
        # Validación si la función es constante
        if isinstance(f, (int, float)) or (
            hasattr(f, "free_symbols") and not f.free_symbols
        ):
            return {
                "x_opt": x_0,
                "f_opt": float(f),
                "path": np.array([x_0]),
                "message": "La función es constante",
            }

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
            step_size = self._get_step_size(
                f_wrapper, grad_f_val, x_k, d_k, t, beta=beta, sigma=sigma, **kwargs
            )

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

        def hess_wrapper(p):
            h = hess_func(p[0], p[1])
            return np.array(h, dtype=float)

        self.hess_wrapper = hess_wrapper

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
            beta_k = np.dot(grad_f_val, grad_f_val) / np.dot(
                self.prev_grad, self.prev_grad
            )
            direction = -grad_f_val + beta_k * self.prev_direction

        self.prev_grad = grad_f_val
        self.prev_direction = direction
        return direction


# --- Restricciones Fáciles ---


class ProjectedGradientStrategy(OptimizationStrategy):
    def optimize(
        self,
        f,
        x_0,
        constraints=None,
        max_iter=1000,
        epsilon=1e-6,
        beta=0.5,
        t=1.0,
        sigma=0.25,
        **kwargs,
    ):
        if constraints is None:
            raise ValueError(
                "Se requieren restricciones de tipo caja para este método."
            )

        x_k = np.array(x_0, dtype=float)

        if not np.allclose(x_k, box_projection(x_k, constraints)):
            raise ValueError(
                r"El punto inicial $x_0$ no cumple las restricciones de caja."
            )

        if isinstance(f, (int, float)) or (
            hasattr(f, "free_symbols") and not f.free_symbols
        ):
            return {
                "x_opt": x_k,
                "f_opt": float(f),
                "path": np.array([x_k]),
                "message": "La función es constante",
            }

        x, y = sp.symbols("x y")
        vars_list = [x, y]

        f_func = sp.lambdify(vars_list, f, "numpy")

        def f_wrapper(p):
            return f_func(p[0], p[1])

        grad_func = get_gradient_func(f, vars_list)

        def grad_wrapper(p):
            g = grad_func(p[0], p[1])
            return np.array(g, dtype=float).flatten()

        path = [x_k.copy()]
        grad_f_val = grad_wrapper(x_k)

        y_k = x_k - grad_f_val
        z_k = box_projection(y_k, constraints)
        d_k = z_k - x_k

        for i in range(max_iter):
            if np.linalg.norm(d_k) < epsilon:
                break

            step_size = armijo_rule(
                f_wrapper, grad_f_val, x_k, d_k, alpha=t, beta=beta, sigma=sigma
            )

            x_new = x_k + step_size * d_k
            grad_f_new = grad_wrapper(x_new)

            y_k = x_new - grad_f_new
            z_k = box_projection(y_k, constraints)
            d_k = z_k - x_new

            x_k = x_new
            grad_f_val = grad_f_new
            path.append(x_k.copy())

        return {
            "x_opt": x_k,
            "f_opt": f_wrapper(x_k),
            "path": np.array(path),
            "message": "Optimización completada (Gradiente Proyectado)",
            "iterations": i + 1,
        }


# --- Restricciones Generales ---
class AugmentedLagrangianStrategy(OptimizationStrategy):
    def optimize(self, f, x_0, constraints=None, max_iter=100, epsilon=1e-6, **kwargs):
        if isinstance(f, (int, float)) or (
            hasattr(f, "free_symbols") and not f.free_symbols
        ):
            x_k = np.array(x_0, dtype=float)
            return {
                "x_opt": x_k,
                "f_opt": float(f),
                "path": np.array([x_k]),
                "message": "La función es constante",
            }

        if constraints is None or not isinstance(constraints, dict):
            raise ValueError("Se requieren restricciones (h y opcionalmente caja).")

        x_k = np.array(x_0, dtype=float)
        h_input = constraints.get("h")
        box_constraints = constraints.get("box")

        if not np.allclose(x_k, box_projection(x_k, box_constraints)):
            raise ValueError(
                r"El punto inicial $x_0$ no cumple las restricciones de caja."
            )

        x, y = sp.symbols("x y")
        vars_list = [x, y]

        # Soportar múltiples restricciones de igualdad
        if isinstance(h_input, str):
            h_strs = [h_input] if h_input.strip() else []
        elif isinstance(h_input, list):
            h_strs = [h for h in h_input if h.strip()]
        else:
            h_strs = []

        if not h_strs:
            raise ValueError("Se requiere al menos una restricción de igualdad h.")

        # Parsear expresiones simbólicas
        h_exprs = []
        for h_str in h_strs:
            try:
                h_exprs.append(sp.sympify(h_str))
            except Exception:
                raise ValueError(f"No se pudo interpretar la restricción h: {h_str}")

        m = len(h_exprs)  # Número de restricciones
        h_funcs = [sp.lambdify(vars_list, h, "numpy") for h in h_exprs]

        # Inicializar multiplicadores de Lagrange (uno por restricción)
        lam = np.zeros(m)
        rho_k = 1.0  # rho_1

        h_prev_norm = float("inf")

        path = [x_k.copy()]

        if box_constraints:
            inner_strategy = ProjectedGradientStrategy()
        else:
            inner_strategy = QuasiNewtonArmijoStrategy()

        for k in range(max_iter):
            # Función Lagrangiano Aumentado para esta iteración
            # L_A = f + sum_i [lam_i * h_i + (rho/2) * h_i^2]
            L_A = f
            for i, h in enumerate(h_exprs):
                L_A = L_A + lam[i] * h + (rho_k / 2) * h**2

            try:
                if box_constraints:
                    res = inner_strategy.optimize(
                        L_A,
                        x_k,
                        constraints=box_constraints,
                        max_iter=100,
                        epsilon=1e-4,
                        **kwargs,
                    )
                else:
                    res = inner_strategy.optimize(
                        L_A, x_k, max_iter=100, epsilon=1e-4, **kwargs
                    )
                x_next = res["x_opt"]
            except Exception as e:
                print(f"Error en optimización interna: {e}")
                break

            # Evaluar todas las restricciones
            h_vals = np.array([h_func(x_next[0], x_next[1]) for h_func in h_funcs])
            h_norm = np.linalg.norm(h_vals)

            if h_norm < epsilon and np.linalg.norm(x_next - x_k) < epsilon:
                x_k = x_next
                path.append(x_k.copy())
                break

            # Actualizar multiplicadores
            lam = lam + rho_k * h_vals

            if h_norm > 0.1 * h_prev_norm:
                rho_k = 10 * rho_k

            h_prev_norm = h_norm
            x_k = x_next
            path.append(x_k.copy())

        f_func = sp.lambdify(vars_list, f, "numpy")
        f_val = f_func(x_k[0], x_k[1])

        return {
            "x_opt": x_k,
            "f_opt": f_val,
            "path": np.array(path),
            "message": "Optimización completada (Lagrangiano Aumentado)",
            "iterations": k + 1,
        }


class PenaltyMethodStrategy(OptimizationStrategy):
    """
    Método de Penalidad con función de penalidad cuadrática.
    Transforma un problema con restricciones en una secuencia de problemas sin restricciones.

    Minimizar f(x)
    s.a. h_i(x) = 0  (restricciones de igualdad)
         g_j(x) <= 0 (restricciones de desigualdad)

    Función penalizada:
    P(x, rho) = f(x) + (rho/2) * [sum_i h_i(x)^2 + sum_j max(0, g_j(x))^2]
    """

    def optimize(
        self,
        f,
        x_0,
        constraints=None,
        max_iter=50,
        epsilon=1e-6,
        beta=0.5,
        sigma=0.25,
        **kwargs,
    ):
        """
        Ejecuta el método de penalidad cuadrática.

        Args:
            f: Función objetivo (expresión sympy)
            x_0: Punto inicial
            constraints: Diccionario con:
                - 'h': lista de restricciones de igualdad h_i(x) = 0
                - 'g': lista de restricciones de desigualdad g_j(x) <= 0
            max_iter: Máximo de iteraciones externas
            epsilon: Tolerancia para factibilidad
        """
        if isinstance(f, (int, float)) or (
            hasattr(f, "free_symbols") and not f.free_symbols
        ):
            x_k = np.array(x_0, dtype=float)
            return {
                "x_opt": x_k,
                "f_opt": float(f),
                "path": np.array([x_k]),
                "message": "La función es constante",
            }

        if constraints is None:
            raise ValueError("Se requieren restricciones para el método de penalidad.")

        x_k = np.array(x_0, dtype=float)
        x, y = sp.symbols("x y")
        vars_list = [x, y]

        # Parsear restricciones de igualdad h(x) = 0
        h_strs = constraints.get("h", [])
        if isinstance(h_strs, str):
            h_strs = [h_strs] if h_strs.strip() else []
        h_strs = [h for h in h_strs if h.strip()]  # Filtrar vacías

        # Parsear restricciones de desigualdad g(x) <= 0
        g_strs = constraints.get("g", [])
        if isinstance(g_strs, str):
            g_strs = [g_strs] if g_strs.strip() else []
        g_strs = [g for g in g_strs if g.strip()]  # Filtrar vacías

        if not h_strs and not g_strs:
            raise ValueError(
                "Se requiere al menos una restricción de igualdad o desigualdad."
            )

        # Parsear expresiones simbólicas
        h_exprs = []
        for h_str in h_strs:
            try:
                h_exprs.append(sp.sympify(h_str))
            except Exception:
                raise ValueError(f"No se pudo interpretar la restricción h: {h_str}")

        g_exprs = []
        for g_str in g_strs:
            try:
                g_exprs.append(sp.sympify(g_str))
            except Exception:
                raise ValueError(f"No se pudo interpretar la restricción g: {g_str}")

        h_funcs = [sp.lambdify(vars_list, h, "numpy") for h in h_exprs]
        g_funcs = [sp.lambdify(vars_list, g, "numpy") for g in g_exprs]

        f_expr = sp.sympify(f) if isinstance(f, str) else f
        f_func = sp.lambdify(vars_list, f_expr, "numpy")

        # Parámetros del método de penalidad
        rho_k = kwargs.get("rho_init", 1.0)
        rho_factor = kwargs.get("rho_factor", 10.0)
        inner_max_iter = kwargs.get("inner_max_iter", 100)

        path = [x_k.copy()]
        inner_strategy = QuasiNewtonArmijoStrategy()
        message = "Se alcanzó el máximo de iteraciones"

        for k in range(max_iter):
            # P(x, rho) = f(x) + (rho/2) * [sum h_i^2 + sum max(0, g_j)^2]
            penalty_term = sp.Integer(0)

            for h in h_exprs:
                penalty_term = penalty_term + h**2
            for g in g_exprs:
                penalty_term = penalty_term + sp.Piecewise((g**2, g > 0), (0, True))

            P = f_expr + (rho_k / 2) * penalty_term

            try:
                res = inner_strategy.optimize(
                    P,
                    x_k,
                    max_iter=inner_max_iter,
                    epsilon=1e-4,
                    beta=beta,
                    sigma=sigma,
                )
                x_next = res["x_opt"]
            except Exception as e:
                try:
                    inner_strategy_backup = GradientDescentStrategy()
                    res = inner_strategy_backup.optimize(
                        P,
                        x_k,
                        max_iter=inner_max_iter,
                        epsilon=1e-4,
                        beta=beta,
                        sigma=sigma,
                    )
                    x_next = res["x_opt"]
                except Exception:
                    message = f"Error en optimización interna: {e}"
                    break

            h_violation = 0.0
            for h_func in h_funcs:
                h_val = h_func(x_next[0], x_next[1])
                h_violation += h_val**2

            g_violation = 0.0
            for g_func in g_funcs:
                g_val = g_func(x_next[0], x_next[1])
                g_violation += max(0, g_val) ** 2

            total_violation = np.sqrt(h_violation + g_violation)

            x_k = x_next
            path.append(x_k.copy())

            if total_violation < epsilon:
                message = "Factibilidad alcanzada"
                break

            rho_k = rho_k * rho_factor

        f_val = f_func(x_k[0], x_k[1])

        return {
            "x_opt": x_k,
            "f_opt": f_val,
            "path": np.array(path),
            "message": f"Optimización completada (Método de Penalidad): {message}",
            "iterations": k + 1,
            "final_rho": rho_k,
        }


class SQPStrategy(OptimizationStrategy):
    def optimize(
        self,
        f,
        x_0,
        constraints=None,
        max_iter=100,
        epsilon=1e-6,
        beta=0.5,
        sigma=0.25,
        **kwargs,
    ):
        """
        Implementation of the Basic SQP Method.
        """
        if isinstance(f, (int, float)) or (
            hasattr(f, "free_symbols") and not f.free_symbols
        ):
            x_k = np.array(x_0, dtype=float)
            return {
                "x_opt": x_k,
                "f_opt": float(f),
                "path": np.array([x_k]),
                "message": "La función es constante",
            }

        x_k = np.array(x_0, dtype=float)
        n = len(x_k)

        if constraints is None or "h" not in constraints:
            raise ValueError(
                "La estrategia SQP requiere restricciones de igualdad 'h'."
            )

        h_input = constraints.get("h")
        if isinstance(h_input, str):
            h_strs = [h_input]
        elif isinstance(h_input, list):
            h_strs = h_input
        else:
            raise ValueError(
                "La restricción 'h' debe ser una cadena o una lista de cadenas."
            )

        m = len(h_strs)
        lam_k = np.zeros(m)

        x, y = sp.symbols("x y")
        vars_list = [x, y]

        f_expr = sp.sympify(f) if isinstance(f, str) else f
        f_func = sp.lambdify(vars_list, f_expr, "numpy")

        grad_f_func = get_gradient_func(f_expr, vars_list)
        hess_f_func = get_hessian(f_expr, vars_list)

        c_exprs = [sp.sympify(h) for h in h_strs]
        c_funcs = [sp.lambdify(vars_list, c, "numpy") for c in c_exprs]

        Jc_sym = [[sp.diff(c, v) for v in vars_list] for c in c_exprs]
        Jc_func = sp.lambdify(vars_list, Jc_sym, "numpy")

        hc_syms = [
            [[sp.diff(c, v1, v2) for v1 in vars_list] for v2 in vars_list]
            for c in c_exprs
        ]
        hc_funcs = [sp.lambdify(vars_list, h_mat, "numpy") for h_mat in hc_syms]

        path = [x_k.copy()]
        message = "Se alcanzó el máximo de iteraciones"

        for k in range(max_iter):
            args = tuple(x_k)

            gf_val = np.array(grad_f_func(*args)).flatten()
            Hf_val = np.array(hess_f_func(*args))

            c_val = np.array([func(*args) for func in c_funcs])
            Jc_val = np.array(Jc_func(*args))
            if m == 1:
                Jc_val = Jc_val.reshape(1, n)

            grad_L = gf_val + Jc_val.T @ lam_k

            B_k = Hf_val.copy()
            for i in range(m):
                Hc_i = np.array(hc_funcs[i](*args))
                B_k += lam_k[i] * Hc_i

            if np.linalg.norm(grad_L) < epsilon and np.linalg.norm(c_val) < epsilon:
                message = "Óptimo encontrado"
                break

            # Form and Solve KKT System
            # [ B_k   Jc^T ] [ d_k ] = [ -grad_L ]
            # [ Jc     0   ] [ xi_k]   [ -c_val  ]

            top_row = np.hstack([B_k, Jc_val.T])
            bot_row = np.hstack([Jc_val, np.zeros((m, m))])
            KKT_matrix = np.vstack([top_row, bot_row])

            rhs_vector = np.concatenate([-grad_L, -c_val])

            try:
                sol = np.linalg.solve(KKT_matrix, rhs_vector)
            except np.linalg.LinAlgError:
                message = "Sistema KKT singular"
                break

            d_k = sol[:n]
            xi_k = sol[n:]

            x_k = x_k + d_k
            lam_k = lam_k + xi_k

            path.append(x_k.copy())

            if np.linalg.norm(x_k) > 1e10:
                message = "El método diverge"
                break

        return {
            "x_opt": x_k,
            "f_opt": f_func(*x_k),
            "path": np.array(path),
            "message": message,
            "iterations": k + 1,
        }
