def graph_contour(title, go, x_range, y_range, Z):
    fig_contour = go.Figure(data=go.Contour(z=Z, x=x_range, y=y_range))
    fig_contour.update_layout(title=title, autosize=False, width=500, height=500)
    return fig_contour

def graph_3d(title, go, x_range, y_range, Z):
    fig_3d = go.Figure(data=[go.Surface(z=Z, x=x_range, y=y_range)])
    fig_3d.update_layout(title=title, autosize=False, width=500, height=500)
    return fig_3d