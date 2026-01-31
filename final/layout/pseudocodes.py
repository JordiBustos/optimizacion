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
        **Datos Iniciales:** $x_0$, $\rho_1 > 0$, $\lambda^{(1)} = (\lambda_1, \dots, \lambda_m) = 0$, $\epsilon$.

        **Lagrangiano Aumentado (múltiples restricciones):**
        $$ L_A(x, \lambda, \rho) = f(x) + \sum_{i=1}^{m} \left[ \lambda_i h_i(x) + \frac{\rho}{2} h_i(x)^2 \right] $$

        **Bucle Iterativo ($k = 1, 2, \dots$):**
        1. **Minimizar Sub-problema:**
           $$ x_k \approx \arg\min_{x} L_A(x, \lambda^{(k)}, \rho_k) $$
        2. **Actualizar Penalización:**
           Si $\|\mathbf{h}(x_k)\| > 0.1 \|\mathbf{h}(x_{k-1})\|$, entonces $\rho_{k+1} = 10 \rho_k$.
           Sino $\rho_{k+1} = \rho_k$.
        3. **Actualizar Multiplicadores:**
           $$ \lambda_i^{(k+1)} = \lambda_i^{(k)} + \rho_k h_i(x_k), \quad \forall i $$
        4. **Parada:** Si $\|\mathbf{h}(x_k)\| < \epsilon$ y convergencia en $x$.
        """
        )


def show_penalty_method_pseudocode():
    with expander("ℹ️ Pseudocódigo"):
        markdown(
            r"""
        **Problema Original:**
        $$ \min f(x) \quad \text{s.a.} \quad h_i(x) = 0, \quad g_j(x) \leq 0 $$

        **Función Penalizada (Cuadrática):**
        $$ P(x, \rho) = f(x) + \frac{\rho}{2} \left[ \sum_i h_i(x)^2 + \sum_j \max(0, g_j(x))^2 \right] $$

        **Datos Iniciales:** $x_0$, $\rho_1 > 0$, $\epsilon$, factor de incremento $c > 1$.

        **Bucle Iterativo ($k = 1, 2, \dots$):**
        1. **Minimizar Sub-problema (sin restricciones):**
           $$ x_k \approx \arg\min_{x} P(x, \rho_k) $$
           (usar método de optimización irrestricta, e.g., Quasi-Newton)
        2. **Verificar Factibilidad:**
           Calcular violación: $\nu_k = \sqrt{\sum_i h_i(x_k)^2 + \sum_j \max(0, g_j(x_k))^2}$
        3. **Criterio de Parada:**
           Si $\nu_k < \epsilon$, terminar.
        4. **Actualizar Penalización:**
           $$ \rho_{k+1} = c \cdot \rho_k $$
           (típicamente $c = 10$)
        """
        )


def show_sqp_pseudocode():
    with expander("ℹ️ Pseudocódigo"):
        markdown(
            r"""
        **Datos Iniciales:** $x_0$, $\lambda_0 = 0$, $\epsilon$.

        **Bucle Iterativo ($k = 0, 1, \dots$):**
        1. **Evaluar:** Gradientes $\nabla f, \nabla c$ y Hessiana del Lagrangiano $B_k = \nabla_{xx}^2 L$.
        2. **Resolver Sistema KKT (Newton):**
           Hallar paso primal $d_k$ y paso dual $\xi_k$ resolviendo:
           $$ \begin{pmatrix} B_k & \nabla c(x_k)^T \\ \nabla c(x_k) & 0 \end{pmatrix} \begin{pmatrix} d_k \\ \xi_k \end{pmatrix} = \begin{pmatrix} -\nabla_x L(x_k, \lambda_k) \\ -c(x_k) \end{pmatrix} $$
        3. **Actualizar:**
           $$ x_{k+1} = x_k + d_k $$,
           $$ \lambda_{k+1} = \lambda_k + \xi_k $$
        4. **Parada:** Si $\|\nabla L(x_{k+1}, \lambda_{k+1})\| < \epsilon$ y $\|c(x_{k+1})\| < \epsilon$.
        """
        )


def show_barrier_method_pseudocode():
    with expander("ℹ️ Pseudocódigo"):
        markdown(
            r"""
        **Problema Original:**
        $$ \min f(x) \quad \text{s.a.} \quad g_j(x) \leq 0, \; j = 1, \dots, p $$

        **Función Barrera Logarítmica:**
        $$ B(x) = \sum_{j=1}^{p} \ln(-g_j(x)) $$

        **Función Penalizada:**
        $$ Q(x, \mu) = f(x) - \mu \cdot B(x) = f(x) - \mu \sum_{j=1}^{p} \ln(-g_j(x)) $$

        **Datos Iniciales:** $x_0$ factible (interior), $\mu_1 > 0$, $\epsilon$, factor de reducción $c > 1$.

        **Bucle Iterativo ($k = 1, 2, \dots$):**
        1. **Minimizar Sub-problema (sin restricciones):**
           $$ x_k \approx \arg\min_{x} Q(x, \mu_k) $$
           (usar método de optimización irrestricta, e.g., Quasi-Newton)
        2. **Criterio de Parada:**
           Si $\|x_k - x_{k-1}\| < \epsilon$, terminar.
        3. **Actualizar Parámetro de Barrera:**
           $$ \mu_{k+1} = \frac{\mu_k}{c} $$
           (típicamente $c = 10$)
        """
        )
