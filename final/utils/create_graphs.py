import plotly.graph_objects as go
import numpy as np


def get_animated_3d_chart(x_range, y_range, Z, path, f_lambdified):
    """
    Generates a self-contained animated 3D figure using Plotly Frames.
    """
    full_z_path = []
    if path is not None and len(path) > 0:
        full_z_path = f_lambdified(path[:, 0], path[:, 1])
        if np.isscalar(full_z_path):
            full_z_path = np.full_like(path[:, 0], full_z_path)

    fig = go.Figure(
        data=[
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
    )

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


def get_animated_contour_chart(x_range, y_range, Z, path, f_lambdified):
    """
    Generates a self-contained animated Contour figure using Plotly Frames.
    """
    fig = go.Figure(
        data=[
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
    )

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
