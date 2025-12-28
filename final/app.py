import streamlit as st
import sympy as sp
import numpy as np
from utils.create_graphs import (
    get_animated_3d_chart,
    get_animated_contour_chart,
)
from strategies import (
    GradientDescentStrategy,
    NewtonStrategy,
    QuasiNewtonArmijoStrategy,
    NonlinearConjugateGradientStrategy,
    ProjectedGradientStrategy,
    AugmentedLagrangianStrategy,
    SQPStrategy,
)
from layout.footer import footer
from layout.sidebar import sidebar
from layout.pseudocodes import (
    show_gradient_descent_pseudocode,
    show_newton_pseudocode,
    show_quasi_newton_pseudocode,
    show_nonlinear_conjugate_gradient_pseudocode,
    show_projected_gradient_pseudocode,
    show_augmented_lagrangian_pseudocode,
    show_sqp_pseudocode,
)


def main():
    st.set_page_config(page_title="Optimización", layout="wide")

    # Inicializar estado de resultados
    if "optimization_result" not in st.session_state:
        st.session_state.optimization_result = None

    sidebar_result = sidebar()
    if sidebar_result is None:
        return

    f, x_0, max_iter, epsilon, beta, sigma, f_lambdified = sidebar_result

    # --- Visualización y Slider ---

    # Generar datos base
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    try:
        Z = f_lambdified(X, Y)
        if np.isscalar(Z):
            Z = np.full_like(X, Z)
    except Exception as e:
        st.error(f"Error al evaluar la función para graficar: {e}")
        Z = np.zeros_like(X)

    # Lógica del Slider y Puntos
    path = None

    if st.session_state.optimization_result:
        path = st.session_state.optimization_result.get("path")
        if path is not None and len(path) > 0:
            st.subheader("Visualización de la Trayectoria")

    col1, col2 = st.columns(2)

    constraints_viz = st.session_state.get("constraints_viz", None)

    with col1:
        fig_contour_animated = get_animated_contour_chart(
            x_range, y_range, Z, path, f_lambdified, constraints=constraints_viz
        )
        st.plotly_chart(fig_contour_animated, use_container_width=True)

    with col2:
        fig_3d_animated = get_animated_3d_chart(
            x_range, y_range, Z, path, f_lambdified, constraints=constraints_viz
        )
        st.plotly_chart(fig_3d_animated, use_container_width=True)

    # --- Selección de Método ---
    st.header("Selección de tipo de optimización")

    constraints = None

    if "category" not in st.session_state:
        st.session_state.category = None

    col_cat1, col_cat2, col_cat3 = st.columns(3)

    with col_cat1:
        if st.button("Optimización Irrestricta"):
            st.session_state.category = "irrestricta"
            st.session_state.constraints_viz = None
            st.rerun()

    with col_cat2:
        if st.button("Restricciones Fáciles"):
            st.session_state.category = "faciles"
            st.rerun()

    with col_cat3:
        if st.button("Restricciones Generales"):
            st.session_state.category = "generales"
            st.session_state.constraints_viz = None
            st.rerun()

    method_name = None
    strategy = None

    # Mapeo de nombres a clases de estrategia
    strategies_map = {
        "Descenso de Gradiente": GradientDescentStrategy,
        "Newton": NewtonStrategy,
        "Quasi-Newton con adaptada BFGS directa": QuasiNewtonArmijoStrategy,
        "Gradientes Conjugados No Lineal": NonlinearConjugateGradientStrategy,
        "Gradiente Proyectado": ProjectedGradientStrategy,
        "Lagrangiano Aumentado": AugmentedLagrangianStrategy,
        "SQP (Programación Cuadrática Secuencial)": SQPStrategy,
    }

    if st.session_state.category == "irrestricta":
        st.subheader("Métodos de Optimización Irrestricta")
        method_name = st.selectbox(
            "Seleccione un método:",
            [
                "Descenso de Gradiente",
                "Newton",
                "Quasi-Newton con adaptada BFGS directa",
                "Gradientes Conjugados No Lineal",
            ],
        )

    elif st.session_state.category == "faciles":
        st.subheader("Optimización con Restricciones Fáciles")
        method_name = st.selectbox("Seleccione un método:", ["Gradiente Proyectado"])
        show_projected_gradient_pseudocode()

        st.markdown("### Definición de la Caja")
        c1, c2 = st.columns(2)
        x_min = c1.number_input("x_min", value=-2.0, key="x_min_input")
        x_max = c2.number_input("x_max", value=2.0, key="x_max_input")
        c3, c4 = st.columns(2)
        y_min = c3.number_input("y_min", value=-2.0, key="y_min_input")
        y_max = c4.number_input("y_max", value=2.0, key="y_max_input")

        constraints = [(x_min, x_max), (y_min, y_max)]
        st.session_state.constraints_viz = constraints

    elif st.session_state.category == "generales":
        st.subheader("Optimización con Restricciones Generales")
        method_name = st.selectbox(
            "Seleccione un método:",
            ["Lagrangiano Aumentado", "SQP (Programación Cuadrática Secuencial)"],
        )

        if method_name == "Lagrangiano Aumentado":
            st.info(
                r"Si no se definen restricciones de caja se utiliza Quasi Newton para la resolución del subproblema, si es provista se utiliza el método de Gradiente Proyectado. El parámetro $\rho$ se ajusta multiplicandolo por $10$ en cada iteración si la norma de la restricción no disminuye adecuadamente. Idealmente $f$ y $h$ deben ser funciones $C^2$."
            )
            show_augmented_lagrangian_pseudocode()
            st.markdown(r"### Restricción de Igualdad $h(x, y) = 0$")
            h_str = st.text_input(r"Ingrese la función $h(x, y)$:", "x + y - 1")

            st.markdown("### Restricciones de Caja (Opcional)")
            use_box = st.checkbox("Usar restricciones de caja", value=True)

            box_constraints = None
            if use_box:
                c1, c2 = st.columns(2)
                x_min = c1.number_input("x_min", value=-5.0, key="x_min_gen")
                x_max = c2.number_input("x_max", value=5.0, key="x_max_gen")
                c3, c4 = st.columns(2)
                y_min = c3.number_input("y_min", value=-5.0, key="y_min_gen")
                y_max = c4.number_input("y_max", value=5.0, key="y_max_gen")
                box_constraints = [(x_min, x_max), (y_min, y_max)]

            constraints = {"h": h_str, "box": box_constraints}
            st.session_state.constraints_viz = constraints

        elif method_name == "SQP (Programación Cuadrática Secuencial)":
            show_sqp_pseudocode()

    if method_name:
        st.write(f"Has seleccionado: **{method_name}**")

        info_c1 = r"Nota: La función debe ser $C^1$ para este método."
        info_c2 = r"Nota: La función debe ser $C^2$ para este método."

        if st.session_state.category == "irrestricta":
            st.info(
                r"Para el step size se utilizan las condiciones de Armijo, el parámetro $\beta$ de la sidebar controla la reducción del mismo y el parámetro $\sigma$ es el de la definición."
            )

        if method_name == "Descenso de Gradiente":
            st.warning(info_c1)
            show_gradient_descent_pseudocode()
        elif method_name == "Newton":
            st.warning(info_c2)
            show_newton_pseudocode()
        elif method_name == "Quasi-Newton con adaptada BFGS directa":
            st.warning(info_c2 + " Se utiliza la identidad como aproximación inicial.")
            show_quasi_newton_pseudocode()
        elif method_name == "Gradientes Conjugados No Lineal":
            st.warning(info_c1)
            st.info(
                r"Para la actualización del parámetro $\beta_k$ de la definición del algoritmo se usa la formula de Fletcher-Reeves. Se utiliza la búsqueda Armijo aunque no garantiza que las direcciones sean de descenso."
            )
            show_nonlinear_conjugate_gradient_pseudocode()

        if st.button("Ejecutar Optimización"):
            strategy_class = strategies_map.get(method_name)
            if strategy_class:
                strategy = strategy_class()
                with st.spinner(f"Ejecutando {method_name}..."):
                    try:
                        result = strategy.optimize(
                            f,
                            x_0,
                            constraints=constraints,
                            max_iter=max_iter,
                            epsilon=epsilon,
                            beta=beta,
                            sigma=sigma,
                        )
                        st.session_state.optimization_result = result
                        st.success("Optimización completada")
                        st.rerun()  # Recargar para mostrar gráficos actualizados
                    except Exception as e:
                        st.error(f"Error durante la optimización: {e}")
            else:
                st.error("Estrategia no implementada.")
    footer()


if __name__ == "__main__":
    main()
