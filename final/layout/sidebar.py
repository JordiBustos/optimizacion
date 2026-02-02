import streamlit as st
import numpy as np
import sympy as sp


def sidebar():
    def reset_all():
        st.session_state.optimization_result = None
        st.session_state.constraints_viz = None
        st.session_state.category = None

    def reset_result_only():
        st.session_state.optimization_result = None

    st.sidebar.header("Definición del Problema")
    func_str = st.sidebar.text_input(
        r"Ingrese la función $f(x, y)$:",
        "(1 - x)**2 + 100 * (y - x**2)**2",
        on_change=reset_all,
    )
    st.sidebar.latex(sp.latex(sp.sympify(func_str)))

    # Input para punto inicial
    st.sidebar.subheader("Punto Inicial")
    x_0_val = st.sidebar.number_input(r"$x_0$", value=-1.2, on_change=reset_result_only)
    y0_val = st.sidebar.number_input(r"$y_0$", value=1.0, on_change=reset_result_only)
    x_0 = np.array([x_0_val, y0_val])

    # Parámetros del algoritmo
    st.sidebar.subheader("Parámetros del Algoritmo")
    max_iter = st.sidebar.number_input(
        "Máximo de iteraciones",
        min_value=1,
        value=100,
        step=10,
        on_change=reset_result_only,
    )
    epsilon = st.sidebar.number_input(
        r"Tolerancia $\epsilon$",
        min_value=1e-10,
        value=1e-6,
        format="%.1e",
        on_change=reset_result_only,
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
            on_change=reset_result_only,
        )
        sigma = st.sidebar.number_input(
            r"$\sigma$",
            min_value=0.01,
            max_value=0.5,
            value=0.25,
            step=0.05,
            on_change=reset_result_only,
        )

    # Rango de visualización del gráfico
    st.sidebar.subheader("Rango de Visualización")
    col_cx, col_cy = st.sidebar.columns(2)
    with col_cx:
        viz_center_x = st.number_input("Centro X", value=0.0, step=0.5, key="viz_center_x")
    with col_cy:
        viz_center_y = st.number_input("Centro Y", value=0.0, step=0.5, key="viz_center_y")
    viz_radius = st.sidebar.number_input("Radio", value=5.0, min_value=0.1, step=1.0, key="viz_radius")
    viz_resolution = st.sidebar.slider(
        "Resolución", min_value=50, max_value=200, value=100, step=10
    )

    try:
        x, y = sp.symbols("x y")
        f = sp.sympify(func_str)
        f_lambdified = sp.lambdify((x, y), f, "numpy")
        st.sidebar.success("Función interpretada correctamente.")
        return (
            f,
            x_0,
            max_iter,
            epsilon,
            beta,
            sigma,
            f_lambdified,
            (viz_center_x, viz_center_y, viz_radius, viz_resolution),
        )
    except Exception as e:
        st.sidebar.error(f"Error al interpretar la función: {e}")
        return None
