# Dynamic Yield Management para Cadenas Hoteleras

Este proyecto aborda uno de los problemas más críticos en la gestión de inventarios perecederos: la pérdida de ingresos por cancelaciones de última hora. Al transformar datos históricos en información predictiva accionable, este modelo permite ejecutar estrategias de *overbooking* seguras, maximizando las ventas sin comprometer la satisfacción del huésped.

## Impacto de Negocio
* **Optimización de Revenue:** Permite sobrevender habitaciones estratégicamente al identificar reservas con >80% de probabilidad de cancelación.
* **Asignación de Recursos:** Ayuda a los equipos de operaciones a prever la ocupación real, ajustando el personal de limpieza y recepción.

## Metodología y Modelamiento
Se evaluó un ensamble de modelos de Machine Learning (Random Forest, AdaBoost, XGBoost) para predecir la variable objetivo `is_canceled`. 
* El modelo ganador fue **XGBoost**, optimizado mediante validación cruzada.
* Se priorizó el área bajo la curva (AUC) y la métrica de *Precision* para minimizar el riesgo operativo de denegar alojamiento a un cliente legítimo.
