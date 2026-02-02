# Proyecto de Optimización

Este proyecto es una aplicación web interactiva construida con Streamlit para visualizar diferentes algoritmos de optimización.

## Requisitos

- Python 3.8+
- Ver `requirements.txt` para las dependencias.

## Instalación

1.  Crea un entorno virtual (opcional pero recomendado):

    ```bash
    python -m venv venv
    source venv/bin/activate  # En Linux/Mac
    # venv\Scripts\activate  # En Windows
    ```

2.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Ejecución

Para iniciar la aplicación, ejecuta el siguiente comando en la terminal desde la carpeta `final`:

```bash
streamlit run app.py
```

## Estructura

- **Optimización Irrestricta**:
  - Descenso de Gradiente
  - Newton
  - Quasi-Newton
  - Gradientes Conjugados No Lineal
- **Restricciones Fáciles**:
  - Gradiente Proyectado
- **Restricciones Generales**:
  - Lagrangiano Aumentado
  - Método de Penalidad
  - Método de barrera
  - SQP
