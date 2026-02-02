from abc import ABC, abstractmethod


class OptimizationStrategy(ABC):
    """
    Clase base abstracta para las estrategias de optimización.
    """

    @abstractmethod
    def optimize(
        self, f, x_0, t=1, max_iter=1000, epsilon=1e-6, sigma=0.25, **kwargs
    ):
        """
        Ejecuta el algoritmo de optimización.

        Args:
            f: La función objetivo (callable o simbólica, según implementación).
            x_0: Punto inicial (numpy array).
            t: Parámetro de paso inicial (float).
            max_iter: Número máximo de iteraciones (int).
            epsilon: Tolerancia para el criterio de parada (float).
            **kwargs: Argumentos adicionales específicos del algoritmo (tolerancia, max_iter, etc.).

        Returns:
            dict: Un diccionario con los resultados, por ejemplo:
                  {'x_opt': punto_optimo, 'f_opt': valor_optimo, 'path': historial_puntos}
        """
        pass
