import numpy as np
from utils.computations import (
    get_gradient_func,
    get_hessian,
    box_projection,
)
from utils.line_search import line_search
from utils.utils import (
    build_algorithm_response,
    make_f_wrapper,
)
import sympy as sp
from .optimization_strategy import OptimizationStrategy
from .easy_constraints import ProjectedGradientStrategy
from .unconstrained import QuasiNewtonArmijoStrategy, GradientDescentStrategy

# Common symbols used across all strategies
_x, _y = sp.symbols("x y")
VARS_LIST = [_x, _y]
SAFE_INFINITY = 1e15
SAFE_BARRIER_DIST = 1e-10


def _parse_constraints(strs, name="h"):
    """Parse constraint strings into sympy expressions and lambdified functions."""
    if isinstance(strs, str):
        strs = [strs] if strs.strip() else []
    strs = [s for s in strs if s.strip()]

    exprs = []
    for s in strs:
        try:
            exprs.append(sp.sympify(s))
        except Exception:
            raise ValueError(f"No se pudo interpretar la restricción {name}: {s}")

    funcs = [sp.lambdify(VARS_LIST, expr, "numpy") for expr in exprs]
    return strs, exprs, funcs


def _eval_at(funcs, point):
    """Evaluate a list of functions at a point."""
    return np.array([f(point[0], point[1]) for f in funcs])


class AugmentedLagrangianStrategy(OptimizationStrategy):
    class_name: str = "Lagrangiano Aumentado"

    def optimize(self, f, x_0, constraints=None, max_iter=100, epsilon=1e-6, **kwargs):
        if constraints is None or not isinstance(constraints, dict):
            raise ValueError("Se requieren restricciones (h y opcionalmente caja).")

        x_k = np.array(x_0, dtype=float)
        box_constraints = constraints.get("box")

        if box_constraints is not None:
            if not np.allclose(x_k, box_projection(x_k, box_constraints)):
                raise ValueError(
                    r"El punto inicial $x_0$ no cumple las restricciones de caja."
                )

        h_strs, h_exprs, h_funcs = _parse_constraints(constraints.get("h", []), "h")
        if not h_strs:
            raise ValueError("Se requiere al menos una restricción de igualdad h.")

        m = len(h_exprs)

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
            h_vals = _eval_at(h_funcs, x_next)
            h_norm = np.linalg.norm(h_vals)

            if h_norm < epsilon or np.linalg.norm(x_next - x_k) < epsilon:
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

        f_func = sp.lambdify(VARS_LIST, f, "numpy")
        return build_algorithm_response(
            x_k, make_f_wrapper(f_func), path, self.class_name, k
        )


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

    class_name: str = "Método de Penalidad"

    def optimize(
        self,
        f,
        x_0,
        constraints=None,
        max_iter=50,
        epsilon=1e-6,
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
        if constraints is None:
            raise ValueError("Se requieren restricciones para el método de penalidad.")

        x_k = np.array(x_0, dtype=float)

        h_strs, h_exprs, h_funcs = _parse_constraints(constraints.get("h", []), "h")
        g_strs, g_exprs, g_funcs = _parse_constraints(constraints.get("g", []), "g")

        if not h_strs and not g_strs:
            raise ValueError(
                "Se requiere al menos una restricción de igualdad o desigualdad."
            )

        f_expr = sp.sympify(f) if isinstance(f, str) else f
        f_func = sp.lambdify(VARS_LIST, f_expr, "numpy")

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
                        sigma=sigma,
                    )
                    x_next = res["x_opt"]
                except Exception:
                    message = f"Error en optimización interna: {e}"
                    break

            h_vals = _eval_at(h_funcs, x_next) if h_funcs else np.array([])
            g_vals = _eval_at(g_funcs, x_next) if g_funcs else np.array([])

            h_violation = np.sum(h_vals**2)
            g_violation = np.sum(np.maximum(0, g_vals) ** 2)

            total_violation = np.sqrt(h_violation + g_violation)

            if np.linalg.norm(x_k - x_next) < epsilon or total_violation < epsilon:
                message = "Factibilidad alcanzada"
                break

            if rho_k > 1e15:
                message = "Parámetro de penalidad demasiado grande"
                break

            x_k = x_next
            path.append(x_k.copy())
            rho_k = rho_k * rho_factor

        return build_algorithm_response(
            x_k, make_f_wrapper(f_func), path, f"{self.class_name}: {message}", k
        )


class SQPStrategy(OptimizationStrategy):
    """
    Método de Programación Cuadrática Secuencial (SQP) para problemas con restricciones de igualdad.
    Minimizar f(x) sujeto a h_i(x) = 0.
    En cada iteración, se resuelve un subproblema cuadrático aproximado.
    """

    class_name: str = "Programación Cuadrática Secuencial"

    def optimize(
        self,
        f,
        x_0,
        constraints=None,
        max_iter=100,
        epsilon=1e-6,
        sigma=0.25,
        sigma_2=0.9,
        **kwargs,
    ):
        """
        Método Básico de Programación Cuadrática Secuencial (SQP).
        """
        x_k = np.array(x_0, dtype=float)
        n = len(x_k)

        if constraints is None or "h" not in constraints:
            raise ValueError(
                "La estrategia SQP requiere restricciones de igualdad 'h'."
            )

        h_strs, c_exprs, c_funcs = _parse_constraints(constraints.get("h", []), "h")
        if not h_strs:
            raise ValueError("Se requiere al menos una restricción de igualdad h.")

        m = len(h_strs)
        lam_k = np.zeros(m)

        f_expr = sp.sympify(f) if isinstance(f, str) else f
        f_func = sp.lambdify(VARS_LIST, f_expr, "numpy")

        grad_f_func = get_gradient_func(f_expr, VARS_LIST)
        hess_f_func = get_hessian(f_expr, VARS_LIST)

        Jc_sym = [[sp.diff(c, v) for v in VARS_LIST] for c in c_exprs]
        Jc_func = sp.lambdify(VARS_LIST, Jc_sym, "numpy")

        hc_syms = [
            [[sp.diff(c, v1, v2) for v1 in VARS_LIST] for v2 in VARS_LIST]
            for c in c_exprs
        ]
        hc_funcs = [sp.lambdify(VARS_LIST, h_mat, "numpy") for h_mat in hc_syms]

        path = [x_k.copy()]
        message = "Se alcanzó el máximo de iteraciones"

        for k in range(max_iter):
            args = tuple(x_k)

            grad_f_val = np.array(grad_f_func(*args), dtype=float).flatten()
            Hf_val = np.array(hess_f_func(*args), dtype=float)

            c_val = np.array([func(*args) for func in c_funcs], dtype=float)
            Jc_val = np.array(Jc_func(*args), dtype=float)
            if m == 1:
                Jc_val = Jc_val.reshape(1, n)

            grad_L = grad_f_val + Jc_val.T @ lam_k

            B_k = Hf_val.copy()
            for i in range(m):
                Hc_i = np.array(hc_funcs[i](*args), dtype=float)
                B_k += lam_k[i] * Hc_i

            # Regularización: si B_k es singular o casi singular, agregar término de identidad
            # Esto es necesario para funciones constantes o casi constantes
            if np.linalg.matrix_rank(B_k) < n:
                B_k = B_k + 1e-6 * np.eye(n)

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

        return build_algorithm_response(
            x_k, make_f_wrapper(f_func), path, f"{self.class_name}: {message}", k
        )


class BarrierMethodStrategy(OptimizationStrategy):
    """
    Método de Barrera para problemas con restricciones de desigualdad.
    Minimizar f(x) sujeto a g_j(x) <= 0.
    Se utiliza una función de barrera logarítmica para mantener las iteraciones dentro de la región factible.
    """

    class_name: str = "Método de Barrera"

    def optimize(
        self,
        f,
        x_0,
        constraints=None,
        max_iter=100,
        epsilon=1e-6,
        sigma=0.25,
        **kwargs,
    ):

        if constraints is None:
            raise ValueError("Se requieren restricciones para el método de barrera.")

        x_k = np.array(x_0, dtype=float)

        g_strs, g_exprs, g_funcs = _parse_constraints(constraints.get("g", []), "g")
        if not g_strs:
            raise ValueError("Se requiere al menos una restricción g(x) <= 0.")

        f_expr = sp.sympify(f)
        f_func = sp.lambdify(VARS_LIST, f_expr, "numpy")

        for i, g_f in enumerate(g_funcs):
            val = g_f(*x_k)
            if val >= -1e-10:
                raise ValueError(
                    f"El punto inicial x_0 viola la restricción {i+1} o está en el borde. "
                    f"Valor: {val}. Se requiere g(x) < 0 estrictamente."
                )

        mu_k = kwargs.get("mu_init", 1.0)
        mu_factor = kwargs.get("mu_factor", 10)
        inner_max_iter = kwargs.get("inner_max_iter", 100)

        path = [x_k.copy()]
        message = "Iteraciones máximas alcanzadas"

        inner_strategy = QuasiNewtonArmijoStrategy()

        for k in range(max_iter):
            # B(x) = - sum( ln( -g_i(x) ) )
            # Q = f + mu * B  =>  Q = f - mu * sum( ln( -g_i(x) ) )

            penalty_sum = sp.Integer(0)
            for g in g_exprs:
                protected_arg = sp.Max(-g, SAFE_BARRIER_DIST)
                safe_log_term = sp.Piecewise(
                    (sp.log(protected_arg), g < -SAFE_BARRIER_DIST),  # Zona segura
                    (-SAFE_INFINITY, True),  # Zona de peligro (Borde/Fuera)
                )
                penalty_sum = penalty_sum + safe_log_term

            Q_expr = f_expr - mu_k * penalty_sum

            try:
                res = inner_strategy.optimize(
                    Q_expr,
                    x_k,
                    variables=VARS_LIST,
                    max_iter=inner_max_iter,
                    epsilon=epsilon,
                    sigma=sigma,
                )
                x_next = res["x_opt"]

                if any(g_f(*x_next) >= -SAFE_BARRIER_DIST for g_f in g_funcs):
                    pass

            except Exception as e:
                message = f"Optimización interna detenida: {e}"
                break

            step_norm = np.linalg.norm(x_next - x_k)
            x_k = x_next
            path.append(x_k.copy())

            if step_norm < epsilon and mu_k < epsilon:
                message = "Convergencia exitosa"
                break

            mu_k = mu_k / mu_factor

        return build_algorithm_response(
            x_k, make_f_wrapper(f_func), path, f"{self.class_name}: {message}", k
        )
