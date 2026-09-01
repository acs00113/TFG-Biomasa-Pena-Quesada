# Modelos

Este directorio contiene el modelo final utilizado en el Trabajo Fin de Grado y
documenta también el modelo de partida empleado durante el ajuste fino.

## 1. Modelo final incluido

El archivo:

`models/best.pt`

corresponde al checkpoint `best.pt` obtenido en el entrenamiento definitivo
realizado mediante GPU/CUDA.

Este es el modelo utilizado posteriormente para realizar la inferencia
territorial sobre el área completa de estudio y generar la comparativa homogénea
con los umbrales de confianza 0,10, 0,15, 0,20 y 0,25.

Configuración principal del entrenamiento definitivo:

- Arquitectura: YOLOv8m
- Tarea: detección
- Clase: `Tree`
- Tamaño de entrada: 1280 px
- Batch: 2
- Épocas: 20
- Ejecución: GPU/CUDA
- GPU empleada: NVIDIA GeForce RTX 3050 Ti Laptop GPU

SHA-256 del modelo final:

`378149D7CAE0C04FDD8EED5BA4FA606DFE72FE8AEA444C26AF8F665BA89EB25E`

Puede comprobarse mediante:

`Get-FileHash .\models\best.pt -Algorithm SHA256`

El script:

`scripts/inference/inferir_comparativa_conf_010_015_020_025.py`

utiliza directamente este archivo.

## 2. Modelo de partida

El modelo utilizado como punto de partida del fine-tuning fue un detector
YOLOv8m preentrenado sobre el conjunto VHRTrees.

Los metadatos internos del checkpoint utilizado permiten identificar la
configuración original:

- Arquitectura: YOLOv8m
- Tarea: detección
- Clase: `Tree`
- Tamaño de entrada: 960 px
- Épocas: 50
- Batch: 16
- Optimizador: `auto`
- Ultralytics registrado: 8.0.196
- Fecha registrada: 2024-04-12
- Correspondencia: VHRTrees Exp-1

SHA-256 del checkpoint baseline utilizado:

`5D4F35C3FB83FE3153A4B2E7395FF10D0A0D3609F3371A97D04000B380C1B08A`

El checkpoint baseline original no se incluye en este repositorio.

## 3. Modelo utilizado para el análisis territorial

El archivo `models/best.pt` es el modelo operativo definitivo del TFG.

La inferencia territorial se realizó con:

- `imgsz = 1280`
- `max_det = 3000`
- IoU NMS = 0,70
- `device = 0`
- `conf = 0,10; 0,15; 0,20; 0,25`

Los resultados congelados de estas ejecuciones se conservan en:

`results/inference/`

De esta forma es posible tanto volver a ejecutar YOLO mediante el modelo
incluido como continuar el procesamiento directamente desde las detecciones
históricas utilizadas en la memoria.
