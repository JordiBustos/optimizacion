import plotly.graph_objects as go
import numpy as np
import sympy as sp


def get_h_points(h_str, x_range, y_range, f_lambdified):
    x, y = sp.symbols("x y")
    try:
        h_expr = sp.sympify(h_str)
        h_func = sp.lambdify((x, y), h_expr, "numpy")
    except:
        return [], [], []

    X, Y = np.meshgrid(x_range, y_range)
    try:
        Z_h = h_func(X, Y)
        if np.isscalar(Z_h):
            Z_h = np.full_like(X, Z_h)
    except:
        return [], [], []

    points_x = []
    points_y = []
    points_z = []

    # Horizontal scan
    for i in range(Z_h.shape[0]):
        for j in range(Z_h.shape[1] - 1):
            v1 = Z_h[i, j]
            v2 = Z_h[i, j + 1]
            if v1 * v2 <= 0:  # Crossing
                if v1 == v2:
                    frac = 0.5
                else:
                    frac = abs(v1) / abs(v1 - v2)

                px = X[i, j] + frac * (X[i, j + 1] - X[i, j])
                py = Y[i, j]
                try:
                    pz = f_lambdified(px, py)
                    points_x.append(px)
                    points_y.append(py)
                    points_z.append(pz)
                except:
                    pass

    # Vertical scan
    for j in range(Z_h.shape[1]):
        for i in range(Z_h.shape[0] - 1):
            v1 = Z_h[i, j]
            v2 = Z_h[i + 1, j]
            if v1 * v2 <= 0:
                if v1 == v2:
                    frac = 0.5
                else:
                    frac = abs(v1) / abs(v1 - v2)

                px = X[i, j]
                py = Y[i, j] + frac * (Y[i + 1, j] - Y[i, j])
                try:
                    pz = f_lambdified(px, py)
                    points_x.append(px)
                    points_y.append(py)
                    points_z.append(pz)
                except:
                    pass

    return points_x, points_y, points_z


def get_animated_3d_chart(x_range, y_range, Z, path, f_lambdified, constraints=None):
    """
    Generates a self-contained animated 3D figure using Plotly Frames.
    """
    full_z_path = []
    if path is not None and len(path) > 0:
        full_z_path = f_lambdified(path[:, 0], path[:, 1])
        if np.isscalar(full_z_path):
            full_z_path = np.full_like(path[:, 0], full_z_path)

    data = [
        go.Surface(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale="Viridis",
            opacity=0.8,
            name="Surface",
            showscale=False,
        ),
        go.Scatter3d(
            x=path[:, 0] if path is not None else [],
            y=path[:, 1] if path is not None else [],
            z=full_z_path,
            mode="lines",
            line=dict(color="white", width=4),
            name="Trayectoria",
        ),
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            marker=dict(color="red", size=5),
            name="Punto Actual",
        ),
    ]

    if constraints:
        box_constraints = None
        h_str = None
        if isinstance(constraints, dict):
            box_constraints = constraints.get("box")
            h_str = constraints.get("h")
        else:
            box_constraints = constraints

        if box_constraints:
            x_min, x_max = box_constraints[0]
            y_min, y_max = box_constraints[1]
            z_min, z_max = np.min(Z), np.max(Z)

            x_box = (
                [x_min, x_max, x_max, x_min, x_min]
                + [None]
                + [x_min, x_max, x_max, x_min, x_min]
                + [None]
                + [x_min, x_min]
                + [None]
                + [x_max, x_max]
                + [None]
                + [x_max, x_max]
                + [None]
                + [x_min, x_min]
            )

            y_box = (
                [y_min, y_min, y_max, y_max, y_min]
                + [None]
                + [y_min, y_min, y_max, y_max, y_min]
                + [None]
                + [y_min, y_min]
                + [None]
                + [y_min, y_min]
                + [None]
                + [y_max, y_max]
                + [None]
                + [y_max, y_max]
            )

            z_box = (
                [z_min, z_min, z_min, z_min, z_min]
                + [None]
                + [z_max, z_max, z_max, z_max, z_max]
                + [None]
                + [z_min, z_max]
                + [None]
                + [z_min, z_max]
                + [None]
                + [z_min, z_max]
                + [None]
                + [z_min, z_max]
            )

            data.append(
                go.Scatter3d(
                    x=x_box,
                    y=y_box,
                    z=z_box,
                    mode="lines",
                    line=dict(color="orange", width=3),
                    name="Restricciones (Caja)",
                )
            )

        if h_str:
            # Soportar tanto string único como lista de strings
            h_list = h_str if isinstance(h_str, list) else [h_str]
            colors_h = ["black", "purple", "brown", "navy", "darkgreen"]
            for idx, h_item in enumerate(h_list):
                if h_item and h_item.strip():
                    hx, hy, hz = get_h_points(h_item, x_range, y_range, f_lambdified)
                    if hx:
                        color = colors_h[idx % len(colors_h)]
                        data.append(
                            go.Scatter3d(
                                x=hx,
                                y=hy,
                                z=hz,
                                mode="markers",
                                marker=dict(color=color, size=2),
                                name=f"h_{idx+1}=0",
                            )
                        )

        # Soportar restricciones de desigualdad g(x,y) <= 0
        g_str = constraints.get("g") if isinstance(constraints, dict) else None
        if g_str:
            g_list = g_str if isinstance(g_str, list) else [g_str]
            colors_g = ["red", "orange", "magenta", "cyan", "yellow"]
            for idx, g_item in enumerate(g_list):
                if g_item and g_item.strip():
                    gx, gy, gz = get_h_points(g_item, x_range, y_range, f_lambdified)
                    if gx:
                        color = colors_g[idx % len(colors_g)]
                        data.append(
                            go.Scatter3d(
                                x=gx,
                                y=gy,
                                z=gz,
                                mode="markers",
                                marker=dict(color=color, size=2),
                                name=f"g_{idx+1}=0",
                            )
                        )

    fig = go.Figure(data=data)

    frames = []
    if path is not None and len(path) > 0:
        for k in range(len(path)):
            curr_point = path[k]
            z_point = f_lambdified(curr_point[0], curr_point[1])

            frames.append(
                go.Frame(
                    data=[
                        go.Scatter3d(x=[curr_point[0]], y=[curr_point[1]], z=[z_point]),
                    ],
                    traces=[2],
                    name=f"frame_{k}",
                    layout=go.Layout(
                        title=f"Iter: {k} | f(x,y)={z_point:.8f} | ({curr_point[0]:.8f}, {curr_point[1]:.8f})"
                    ),
                )
            )

    fig.frames = frames

    fig.update_layout(
        title="Optimización 3D",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
        margin=dict(b=50),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=50, redraw=True), fromcurrent=True
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [f"frame_{k}"],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": str(k),
                    }
                    for k in range(len(frames))
                ],
                "currentvalue": {"prefix": "Iteración: "},
                "pad": {"t": 50},
            }
        ],
    )

    if path is not None and len(path) > 0:
        z_start = f_lambdified(path[0, 0], path[0, 1])
        fig.update_traces(
            x=[path[0, 0]],
            y=[path[0, 1]],
            z=[z_start],
            selector=dict(name="Punto Actual"),
        )
        fig.update_layout(
            title=f"Iter: 0 | f(x,y)={z_start:.8f} | ({path[0,0]:.8f}, {path[0,1]:.8f})"
        )

    return fig


def get_animated_contour_chart(
    x_range, y_range, Z, path, f_lambdified, constraints=None
):
    """
    Generates a self-contained animated Contour figure using Plotly Frames.
    """
    data = [
        go.Contour(z=Z, x=x_range, y=y_range, name="Contour"),
        go.Scatter(
            x=path[:, 0] if path is not None else [],
            y=path[:, 1] if path is not None else [],
            mode="lines",
            line=dict(color="white", width=2),
            name="Trayectoria",
        ),
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Punto Actual",
        ),
    ]

    if constraints:
        box_constraints = None
        h_str = None
        if isinstance(constraints, dict):
            box_constraints = constraints.get("box")
            h_str = constraints.get("h")
        else:
            box_constraints = constraints

        if box_constraints:
            x_min, x_max = box_constraints[0]
            y_min, y_max = box_constraints[1]

            data.append(
                go.Scatter(
                    x=[x_min, x_max, x_max, x_min, x_min],
                    y=[y_min, y_min, y_max, y_max, y_min],
                    mode="lines",
                    line=dict(color="orange", width=2, dash="dash"),
                    name="Restricciones (Caja)",
                )
            )

        if h_str:
            # Soportar tanto string único como lista de strings
            h_list = h_str if isinstance(h_str, list) else [h_str]
            colors_h = ["black", "purple", "brown", "navy", "darkgreen"]
            for idx, h_item in enumerate(h_list):
                if h_item and h_item.strip():
                    hx, hy, _ = get_h_points(h_item, x_range, y_range, f_lambdified)
                    if hx:
                        color = colors_h[idx % len(colors_h)]
                        data.append(
                            go.Scatter(
                                x=hx,
                                y=hy,
                                mode="markers",
                                marker=dict(color=color, size=2),
                                name=f"h_{idx+1}=0",
                            )
                        )

        # Soportar restricciones de desigualdad g(x,y) <= 0
        g_str = constraints.get("g") if isinstance(constraints, dict) else None
        if g_str:
            g_list = g_str if isinstance(g_str, list) else [g_str]
            colors_g = ["red", "orange", "magenta", "cyan", "yellow"]
            for idx, g_item in enumerate(g_list):
                if g_item and g_item.strip():
                    gx, gy, _ = get_h_points(g_item, x_range, y_range, f_lambdified)
                    if gx:
                        color = colors_g[idx % len(colors_g)]
                        data.append(
                            go.Scatter(
                                x=gx,
                                y=gy,
                                mode="markers",
                                marker=dict(color=color, size=2),
                                name=f"g_{idx+1}=0",
                            )
                        )

    fig = go.Figure(data=data)

    frames = []
    if path is not None and len(path) > 0:
        for k in range(len(path)):
            curr_point = path[k]
            z_point = f_lambdified(curr_point[0], curr_point[1])

            frames.append(
                go.Frame(
                    data=[
                        go.Scatter(x=[curr_point[0]], y=[curr_point[1]]),
                    ],
                    traces=[2],
                    name=f"frame_{k}",
                    layout=go.Layout(
                        title=f"Iter: {k} | f(x,y)={z_point:.8f} | ({curr_point[0]:.8f}, {curr_point[1]:.8f})"
                    ),
                )
            )

    fig.frames = frames

    fig.update_layout(
        title="Gráfico de Nivel",
        xaxis_title="X",
        yaxis_title="Y",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=50),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=50, redraw=True), fromcurrent=True
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [f"frame_{k}"],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": str(k),
                    }
                    for k in range(len(frames))
                ],
                "currentvalue": {"prefix": "Iteración: "},
                "pad": {"t": 50},
            }
        ],
    )

    if path is not None and len(path) > 0:
        z_start = f_lambdified(path[0, 0], path[0, 1])
        fig.update_traces(
            x=[path[0, 0]], y=[path[0, 1]], selector=dict(name="Punto Actual")
        )
        fig.update_layout(
            title=f"Iter: 0 | f(x,y)={z_start:.4f} | ({path[0,0]:.4f}, {path[0,1]:.4f})"
        )

    return fig


def get_convergence_chart(path, f_lambdified):
    """
    Generates a line chart of f(x_k) vs iteration k.
    """
    if path is None or len(path) == 0:
        return None

    f_values = []
    for p in path:
        try:
            val = f_lambdified(p[0], p[1])
            f_values.append(val)
        except:
            f_values.append(None)

    iterations = list(range(len(f_values)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=f_values,
            mode="lines+markers",
            name="f(x_k)",
            line=dict(color="blue"),
        )
    )

    fig.update_layout(
        title="Convergencia de f(x)",
        xaxis_title="Iteración (k)",
        yaxis_title="f(x_k)",
        template="plotly_white",
    )

    return fig
