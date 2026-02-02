import numpy as np


def build_algorithm_response(x_k, f_wrapper, path, name, i):
    return {
        "x_opt": x_k,
        "f_opt": f_wrapper(x_k),
        "path": np.array(path),
        "message": f"Optimización completada {name}",
        "iterations": i + 1,
    }


def make_f_wrapper(f_func):
    """Create a wrapper function for f that takes a point array."""
    return lambda p: f_func(p[0], p[1])
