from streamlit import expander, markdown


def show_gradient_descent_pseudocode():
    with expander("ℹ️ Pseudocódigo"):
        markdown(
            r"""
        **Datos Iniciales:** $x_0$, $\epsilon$, parámetros de Armijo ($\beta, \sigma$).

        **Bucle Iterativo ($k = 0, 1, \dots$):**
        1. Calcular dirección de descenso: $d_k = -\nabla f(x_k)$.
        2. Determinar paso $\alpha_k$ mediante la regla de Armijo.
        3. Actualizar: $x_{k+1} = x_k + \alpha_k d_k$.
        4. **Parada:** Si $\|\nabla f(x_{k+1})\| < \epsilon$.
        """
        )


def show_newton_pseudocode():
    with expander("ℹ️ Pseudocódigo"):
        markdown(
            r"""
        **Datos Iniciales:** $x_0$, $\epsilon$.

        **Bucle Iterativo ($k = 0, 1, \dots$):**
        1. Calcular Hessiano $\nabla^2 f(x_k)$ y Gradiente $\nabla f(x_k)$.
        2. Resolver sistema: $\nabla^2 f(x_k) d_k = -\nabla f(x_k)$.
        3. Determinar paso $\alpha_k$ utilizando la regla de Armijo.
        4. Actualizar: $x_{k+1} = x_k + \alpha_k d_k$.
        5. **Parada:** Si $\|\nabla f(x_{k+1})\| < \epsilon$.
        """
        )


def show_quasi_newton_pseudocode():
    with expander("ℹ️ Pseudocódigo"):
        markdown(
            r"""
        **Datos Iniciales:** $x_0$, $B_0 = I$, $\epsilon$.

        **Bucle Iterativo ($k = 0, 1, \dots$):**
        1. Calcular dirección: $B_k d_k = -\nabla f(x_k)$.
        2. Determinar paso $\alpha_k$ mediante la regla de Armijo.
        3. Actualizar: $x_{k+1} = x_k + \alpha_k d_k$.
        4. Calcular $s_k = x_{k+1} - x_k$ y $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$.
        5. Actualizar $B_{k+1}$ usando fórmula BFGS.
        6. **Parada:** Si $\|\nabla f(x_{k+1})\| < \epsilon$.
        """
        )


def show_nonlinear_conjugate_gradient_pseudocode():
    with expander("ℹ️ Pseudocódigo"):
        markdown(
            r"""
        **Datos Iniciales:** $x_0$, $\epsilon$.
        $d_0 = -\nabla f(x_0)$.

        **Bucle Iterativo ($k = 0, 1, \dots$):**
        1. Determinar paso $\alpha_k$ (Armijo).
        2. Actualizar: $x_{k+1} = x_k + \alpha_k d_k$.
        3. Calcular $\beta_{k+1}$ (Fletcher-Reeves)
        4. Actualizar dirección: $d_{k+1} = -\nabla f(x_{k+1}) + \beta_{k+1} d_k$.
        5. **Parada:** Si $\|\nabla f(x_{k+1})\| < \epsilon$.
        """
        )


def show_projected_gradient_pseudocode():
    with expander("ℹ️ Pseudocódigo"):
        markdown(
            r"""
        **Datos Iniciales:** $x_0 \in \Omega$, $\epsilon$.

        **Bucle Iterativo ($k = 0, 1, \dots$):**
        1. Calcular dirección proyectada:
           $$ d_k = P_{\Omega}(x_k - \nabla f(x_k)) - x_k $$
        2. Determinar paso $\alpha_k$ (Armijo) para $f(x_k + \alpha d_k)$.
        3. Actualizar: $x_{k+1} = x_k + \alpha_k d_k$.
        4. **Parada:** Si $\|d_k\| < \epsilon$.
        """
        )


def show_augmented_lagrangian_pseudocode():
    with expander("ℹ️ Pseudocódigo"):
        markdown(
            r"""
        **Datos Iniciales:** $x_0$, $\rho_1 > 0$, $\lambda_1$, $\epsilon$.

        **Bucle Iterativo ($k = 1, 2, \dots$):**
        1. **Minimizar Sub-problema:**
           $$ x_k \approx \arg\min_{x} L_A(x, \lambda_k, \rho_k) = f(x) + \lambda_k h(x) + \frac{\rho_k}{2} h(x)^2 $$
        2. **Actualizar Penalización:**
           Si $\|h(x_k)\| > 0.1 \|h(x_{k-1})\|$, entonces $\rho_{k+1} = 10 \rho_k$.
           Sino $\rho_{k+1} = \rho_k$.
        3. **Actualizar Multiplicadores:**
           $$ \lambda_{k+1} = \lambda_k + \rho_k h(x_k) $$
        4. **Parada:** Si $\|h(x_k)\| < \epsilon$ y convergencia en $x$.
        """
        )


def show_sqp_pseudocode():
    with expander("ℹ️ Pseudocódigo"):
        markdown(
            r"""
        TODO 
        """
        )
