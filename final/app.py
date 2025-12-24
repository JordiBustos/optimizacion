import streamlit as st
import sympy as sp
import numpy as np
import time
import plotly.graph_objects as go
from utils.create_graphs import graph_contour, graph_3d
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


def main():
    st.set_page_config(page_title="Optimización", layout="wide")

    # Inicializar estado de resultados
    if "optimization_result" not in st.session_state:
        st.session_state.optimization_result = None

    def reset_state():
        st.session_state.optimization_result = None

    st.sidebar.header("Definición del Problema")
    func_str = st.sidebar.text_input(
        r"Ingrese la función $f(x, y)$:",
        "(1 - x)**2 + 100 * (y - x**2)**2",
        on_change=reset_state,
    )

    # Input para punto inicial
    st.sidebar.subheader("Punto Inicial")
    x0_val = st.sidebar.number_input(r"$x_0$", value=-1.2, on_change=reset_state)
    y0_val = st.sidebar.number_input(r"$y_0$", value=1.0, on_change=reset_state)
    x0 = np.array([x0_val, y0_val])

    # Parámetros del algoritmo
    st.sidebar.subheader("Parámetros del Algoritmo")
    max_iter = st.sidebar.number_input(
        "Máximo de iteraciones", min_value=1, value=100, step=10, on_change=reset_state
    )
    epsilon = st.sidebar.number_input(
        r"Tolerancia $\epsilon$",
        min_value=1e-10,
        value=1e-6,
        format="%.1e",
        on_change=reset_state,
    )

    beta = 0.5
    sigma = 0.25

    if st.session_state.get("category") == "irrestricta":
        st.sidebar.subheader("Parámetros de Armijo")
        beta = st.sidebar.number_input(
            r"$\beta$ (factor de reducción)",
            min_value=0.01,
            max_value=0.99,
            value=0.5,
            step=0.05,
            on_change=reset_state,
        )
        sigma = st.sidebar.number_input(
            r"$\sigma$",
            min_value=0.01,
            max_value=0.5,
            value=0.25,
            step=0.05,
            on_change=reset_state,
        )

    try:
        x, y = sp.symbols("x y")
        f = sp.sympify(func_str)
        f_lambdified = sp.lambdify((x, y), f, "numpy")
        st.sidebar.success("Función interpretada correctamente.")
    except Exception as e:
        st.sidebar.error(f"Error al interpretar la función: {e}")
        return

    # --- Visualización y Slider ---

    # Generar datos base
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    try:
        Z = f_lambdified(X, Y)
    except Exception as e:
        st.error(f"Error al evaluar la función para graficar: {e}")
        Z = np.zeros_like(X)

    # Lógica del Slider y Puntos
    current_point = None
    path = None
    iteration = 0
    animate = False

    if st.session_state.optimization_result:
        path = st.session_state.optimization_result.get("path")
        if path is not None and len(path) > 0:
            st.subheader("Visualización de la Trayectoria")

            col_slider, col_btn = st.columns([4, 1])

            with col_btn:
                st.write("")  # Espaciado para alinear con el slider
                st.write("")
                if st.button("▶️ Animar"):
                    animate = True

            with col_slider:
                if len(path) > 1:
                    iteration = st.slider("Iteración", 0, len(path) - 1, 0)
                else:
                    iteration = 0

            current_point = path[iteration]
            st.write(
                f"Iteración: {iteration}, Punto: {current_point}, Valor: {f_lambdified(current_point[0], current_point[1]):.4f}"
            )

    col1, col2 = st.columns(2)

    with col1:
        plot_spot_contour = st.empty()
    with col2:
        plot_spot_3d = st.empty()

    def render_plots(idx):
        # Crear figuras base
        fig_contour = graph_contour(
            "Gráfico de Nivel (Contour)", go, x_range, y_range, Z
        )
        fig_3d = graph_3d("Gráfico 3D (Surface)", go, x_range, y_range, Z)

        # Añadir trazas si hay resultados
        if path is not None and len(path) > 0:
            # Asegurar índice válido
            safe_idx = min(idx, len(path) - 1)
            curr_p = path[safe_idx]

            # Contour
            fig_contour.add_trace(
                go.Scatter(
                    x=path[:, 0],
                    y=path[:, 1],
                    mode="lines",
                    line=dict(color="white", width=2),
                    name="Trayectoria",
                )
            )
            fig_contour.add_trace(
                go.Scatter(
                    x=[curr_p[0]],
                    y=[curr_p[1]],
                    mode="markers",
                    marker=dict(color="red", size=10),
                    name="Punto Actual",
                )
            )

            # 3D
            z_path = f_lambdified(path[:, 0], path[:, 1])
            z_point = f_lambdified(curr_p[0], curr_p[1])

            fig_3d.add_trace(
                go.Scatter3d(
                    x=path[:, 0],
                    y=path[:, 1],
                    z=z_path,
                    mode="lines",
                    line=dict(color="white", width=4),
                    name="Trayectoria",
                )
            )
            fig_3d.add_trace(
                go.Scatter3d(
                    x=[curr_p[0]],
                    y=[curr_p[1]],
                    z=[z_point],
                    mode="markers",
                    marker=dict(color="red", size=5),
                    name="Punto Actual",
                )
            )

        plot_spot_contour.plotly_chart(fig_contour)
        plot_spot_3d.plotly_chart(fig_3d)

    if animate and path is not None and len(path) > 1:
        for i in range(len(path)):
            render_plots(i)
            time.sleep(0.1)  # Velocidad de animación
    else:
        render_plots(iteration)

    # --- Selección de Método ---
    st.header("Selección de tipo de optimización")

    if "category" not in st.session_state:
        st.session_state.category = None

    col_cat1, col_cat2, col_cat3 = st.columns(3)

    with col_cat1:
        if st.button("Optimización Irrestricta"):
            st.session_state.category = "irrestricta"
            st.rerun()

    with col_cat2:
        if st.button("Restricciones Fáciles"):
            st.session_state.category = "faciles"
            st.rerun()

    with col_cat3:
        if st.button("Restricciones Generales"):
            st.session_state.category = "generales"
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

    elif st.session_state.category == "generales":
        st.subheader("Optimización con Restricciones Generales")
        method_name = st.selectbox(
            "Seleccione un método:",
            ["Lagrangiano Aumentado", "SQP (Programación Cuadrática Secuencial)"],
        )

    if method_name:
        st.write(f"Has seleccionado: **{method_name}**")

        info_c1 = r"Nota: La función debe ser $C^1$ para este método."
        info_c2 = r"Nota: La función debe ser $C^2$ para este método."

        if st.session_state.category == "irrestricta":
            st.info(
                "Para el step size se utilizan las condiciones de Armijo. El parámetro Beta de la sidebar controla la reducción del mismo y el parámetro sigma es el de la definición."
            )

        if method_name == "Descenso de Gradiente":
            st.warning(info_c1)
        elif method_name == "Newton":
            st.warning(info_c2)
        elif method_name == "Quasi-Newton con adaptada BFGS directa":
            st.warning(info_c2)

        if st.button("Ejecutar Optimización"):
            strategy_class = strategies_map.get(method_name)
            if strategy_class:
                strategy = strategy_class()
                with st.spinner(f"Ejecutando {method_name}..."):
                    try:
                        result = strategy.optimize(
                            f,
                            x0,
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
