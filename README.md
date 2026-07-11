# 💧 Dashboard — Detección de Fugas SEDALIB S.A.

Dashboard interactivo desarrollado como parte del Proyecto Final del curso
**Cómputo Distribuido y Paralelo*

## Descripción

El proyecto implementa una solución distribuida de Machine Learning para la
detección temprana de fugas de agua en la red de distribución de SEDALIB S.A.,
mediante un modelo Random Forest entrenado bajo una arquitectura maestro–trabajador
con paralelismo de datos, tolerancia a fallos y monitoreo del rendimiento.

Este dashboard consolida los resultados del proyecto en cinco secciones:

- **📊 Insights**: importancia de variables y ranking de zonas de riesgo.
- **⚡ Rendimiento**: comparación de tiempos secuencial vs. paralelo vs. distribuido.
- **🛡️ Tolerancia a fallos**: simulación en vivo de caída y recuperación de un nodo.
- **🖥️ Monitoreo**: matriz de confusión y métricas de clasificación del modelo.
- **🚨 Simulador**: clasificación de una nueva lectura de sensor en tiempo real.

## Tecnologías

- Python, scikit-learn (Random Forest)
- Streamlit (interfaz del dashboard)
- Google Colab (entrenamiento y experimentación)

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nota

El dataset utilizado es simulado, generado a partir de patrones hidráulicos
documentados, debido a que la telemetría real de SEDALIB S.A. no es de acceso
público.
