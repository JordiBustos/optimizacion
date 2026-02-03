import numpy as np
from utils.computations import (
    get_gradient_func,
    is_f_constant,
    box_projection,
)
from utils.line_search import line_search
from utils.utils import build_algorithm_response
import sympy as sp
from .optimization_strategy import OptimizationStrategy


class ProjectedGradientStrategy(OptimizationStrategy):
    class_name = "Gradiente Proyectado"
    x, y = sp.symbols("x y")
    vars_list = [x, y]

    def optimize(
        self,
        f,
        x_0,
        constraints=None,
        max_iter=1000,
        epsilon=1e-6,
        t=1.0,
        sigma=0.25,
        sigma_2=0.9,
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

        is_constant = is_f_constant(f, x_0)
        if is_constant["is_constant"]:
            return is_constant

        f_func = sp.lambdify(self.vars_list, f, "numpy")

        def f_wrapper(p):
            return f_func(p[0], p[1])

        grad_func = get_gradient_func(f, self.vars_list)

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

            step_size = line_search(
                f_wrapper,
                grad_f_val,
                x_k,
                d_k,
                alpha=t,
                sigma=sigma,
                grad_wrapper=grad_wrapper,
                sigma_2=sigma_2,
                amax=1.0,
            )

            x_new = x_k + step_size * d_k

            # Proyectar nuevamente para asegurar factibilidad por errores de redondeo
            x_new = box_projection(x_new, constraints)

            grad_f_new = grad_wrapper(x_new)

            y_k = x_new - grad_f_new
            z_k = box_projection(y_k, constraints)
            d_k = z_k - x_new

            x_k = x_new
            grad_f_val = grad_f_new
            path.append(x_k.copy())

        return build_algorithm_response(x_k, f_wrapper, path, self.class_name, i)
