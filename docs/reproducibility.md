# Reproducibilidad

Este documento describe los distintos niveles de reproducción posibles a partir del material conservado en el repositorio del Trabajo Fin de Grado.

La estructura distingue entre:

- `results/`: resultados congelados utilizados como referencia en la memoria.
- `outputs/`: resultados generados al volver a ejecutar los scripts.

Los scripts de reproducción no deben sobrescribir los resultados publicados en `results/`.

## 1. Evaluación del detector

Esta fase puede reproducirse directamente con los archivos incluidos en el repositorio.

Entradas:

- `data/dataset/labels/val/`
- `data/evaluation_predictions/`

No requiere pesos del modelo, datos PNOA territoriales, LiDAR ni GDAL.

Ejecución:

`python .\scripts\evaluation\evaluar_tp_fp_fn.py`

Las salidas se generan en:

`outputs/evaluation/`

Posteriormente puede ejecutarse:

`python .\scripts\evaluation\analizar_resultados_tp_fp_fn.py`

Durante la preparación final del repositorio se comprobó que los seis CSV principales regenerados son idénticos a los resultados congelados en sus campos científicos. Las únicas diferencias esperables corresponden a campos que almacenan rutas locales dependientes de la máquina.

Los resultados congelados de referencia se encuentran en:

`results/detector_evaluation/`

## 2. Regeneración de figuras

Las figuras correspondientes al análisis de biomasa y variabilidad espacial pueden regenerarse a partir de los resultados congelados sin disponer del modelo YOLO, PNOA o LiDAR.

Ejecución:

`python .\scripts\visualization\generar_visualizaciones_biomasa_por_recorte_AJUSTADO.py`

El script utiliza por defecto:

- `results/spatial_analysis/comparacion_biomasa_por_recorte.csv`
- `results/biomass/resumen_global_comparativa_biomasa.csv`

Las figuras se generan en:

`outputs/figures/`

Esta ejecución fue comprobada satisfactoriamente durante la preparación del repositorio.

## 3. Georreferenciación desde detecciones congeladas

El repositorio conserva las detecciones territoriales correspondientes a la comparativa homogénea de los cuatro umbrales:

- `results/inference/conf010/labels/`
- `results/inference/conf015/labels/`
- `results/inference/conf020/labels/`
- `results/inference/conf025/labels/`

La rejilla territorial consta de 180 posiciones. Un experimento puede contener 179 archivos TXT si una posición no produjo ninguna detección.

Para reproducir la georreferenciación también son necesarios:

- `data/external/pnoa_tiles/geotiff/`
- `data/external/lidar/Altura_Árboles_Biomasa.tif`

Su procedencia y características se documentan en `data/external/README.md`.

En Windows, utilizando el entorno QGIS empleado durante el TFG:

`& "C:\Program Files\QGIS 3.44.9\bin\python-qgis-ltr.bat" ".\scripts\georeferencing\exportar_detecciones_lidar_comparativa_conf.py"`

Las salidas se generan en:

`outputs/georeferencing/`

## 4. Cálculo de biomasa

Una vez generadas las salidas de georreferenciación puede ejecutarse:

`& "C:\Program Files\QGIS 3.44.9\bin\python-qgis-ltr.bat" ".\scripts\biomass\calcular_biomasa_por_recorte_comparativa_conf.py"`

Las salidas se generan en:

`outputs/biomass/`

El flujo utiliza el raster de altura:

`data/external/lidar/Altura_Árboles_Biomasa.tif`

SHA-256:

`177FB042B1CEBC19D2E9C7FBD7C251C1B80D2AB1AD00CEA35D8E8B88C221FD17`

## 5. Flujo territorial completo desde YOLO

Para comenzar desde la inferencia son necesarios:

- `models/best.pt`
- `data/external/pnoa_tiles/png/`
- `data/external/pnoa_tiles/geotiff/`
- `data/external/lidar/Altura_Árboles_Biomasa.tif`

La documentación del modelo se encuentra en `models/README.md`.

La inferencia homogénea puede ejecutarse mediante:

`& "C:\conda_envs\tfg_yolo_cuda\python.exe" ".\scripts\inference\inferir_comparativa_conf_010_015_020_025.py"`

La configuración empleada en el TFG fue:

- `imgsz = 1280`
- `max_det = 3000`
- `IoU NMS = 0.70`
- `device = 0`
- `conf = 0.10, 0.15, 0.20, 0.25`

Las nuevas etiquetas se almacenan en `outputs/inference/`.

Para continuar la cadena utilizando estas detecciones regeneradas:

`$env:TFG_INFERENCE_DIR = (Resolve-Path ".\outputs\inference").Path`

y posteriormente:

`& "C:\Program Files\QGIS 3.44.9\bin\python-qgis-ltr.bat" ".\scripts\georeferencing\exportar_detecciones_lidar_comparativa_conf.py"`

Al finalizar puede eliminarse la variable temporal mediante:

`Remove-Item Env:TFG_INFERENCE_DIR -ErrorAction SilentlyContinue`

## 6. Modelo final

El modelo operativo utilizado en la comparativa territorial homogénea está identificado mediante:

SHA-256:

`378149D7CAE0C04FDD8EED5BA4FA606DFE72FE8AEA444C26AF8F665BA89EB25E`

Debe situarse como:

`models/best.pt`

El checkpoint final se incluye directamente en el repositorio como `models/best.pt`. La procedencia, configuración e identificadores de los modelos se documentan en `models/README.md`.

## 7. Entrenamiento

El repositorio conserva evidencias del entrenamiento definitivo en:

`training/final_gpu/`

El modelo final procede del ajuste fino de un YOLOv8m preentrenado sobre VHRTrees utilizando el conjunto propio de Peña de Quesada.

La configuración principal fue:

- `imgsz = 1280`
- `batch = 2`
- `epochs = 20`
- clase única: `Tree`

El entrenamiento histórico completo no se presenta como una cadena automatizada desde cero. La reproducción principal se centra en las fases para las que se conservan explícitamente scripts, entradas y resultados.

## 8. Entornos de ejecución

Las versiones verificadas de Python, PyTorch, Ultralytics, QGIS y GDAL se documentan en:

`docs/environment.md`

El proyecto utilizó entornos diferenciados para:

- evaluación y visualización;
- inferencia YOLO con CUDA;
- procesamiento geoespacial mediante QGIS/GDAL.

Esta separación reproduce de forma más fiel el flujo real del proyecto que un único entorno artificialmente unificado.

