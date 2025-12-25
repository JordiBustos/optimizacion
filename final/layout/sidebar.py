import streamlit as st
import numpy as np
import sympy as sp


def sidebar():
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
    x_0_val = st.sidebar.number_input(r"$x_0$", value=-1.2, on_change=reset_state)
    y0_val = st.sidebar.number_input(r"$y_0$", value=1.0, on_change=reset_state)
    x_0 = np.array([x_0_val, y0_val])

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
        return f, x_0, max_iter, epsilon, beta, sigma, f_lambdified
    except Exception as e:
        st.sidebar.error(f"Error al interpretar la función: {e}")
        return None
