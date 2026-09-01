# Ejemplos visuales de detección

Este directorio conserva ejemplos visuales representativos de la detección de
arbolado realizada con el modelo final del Trabajo Fin de Grado.

Los ejemplos proceden del checkpoint definitivo:

`models/best.pt`

y corresponden al umbral:

`conf = 0,25`

## Casos representados

Se incluye una imagen de cada nivel de densidad utilizado durante la preparación
y evaluación visual del detector:

- alta densidad;
- densidad media;
- baja densidad.

Las imágenes utilizadas como referencia son:

- `Alta-densidad_02`
- `Densidad_media_07`
- `Baja_densidad_03`

Estas imágenes pertenecen a la partición de validación del conjunto local
descrito en `data/dataset/README.md`.

## Versiones incluidas

### blue_labels

Contiene la salida visual generada por YOLO, mostrando las detecciones mediante
cajas, clase y confianza asociada.

Archivos:

- `alta_densidad_blue`
- `densidad_media_blue`
- `baja_densidad_blue`

### yellow_boxes

Contiene una representación alternativa de las mismas detecciones mediante cajas
amarillas sin etiquetas de texto.

El repositorio incluye además el script:

`scripts/visualization/pintar-cajas-amarillo-comparativa-conf.py`

para generar este tipo de representación sobre las detecciones territoriales.

Archivos:

- `alta_densidad_yellow`
- `densidad_media_yellow`
- `baja_densidad_yellow`

## Finalidad

Estas imágenes no sustituyen a las etiquetas YOLO ni a los resultados numéricos
del experimento.

Su finalidad es proporcionar una evidencia visual inmediata del comportamiento
del modelo final en escenarios de diferente densidad arbórea y facilitar la
interpretación del repositorio sin necesidad de ejecutar previamente el modelo.

Las predicciones utilizadas para la evaluación cuantitativa se conservan por
separado en:

`data/evaluation_predictions/`
