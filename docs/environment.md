# Entornos de ejecución

El flujo completo de este Trabajo Fin de Grado combina tareas de detección
mediante aprendizaje profundo, análisis tabular y representación gráfica, y
procesamiento geoespacial mediante GDAL.

Durante el desarrollo y la preparación del repositorio se utilizaron y
verificaron tres entornos diferenciados. Se documentan por separado para no
presentar artificialmente todas las dependencias como si hubieran pertenecido
a una única instalación de Python.

## 1. Entorno de inferencia YOLO con GPU

Este entorno se utilizó para la inferencia territorial con el modelo final y
dispone de PyTorch con soporte CUDA.

Configuración verificada:

- Python: 3.10.20
- Distribución de Python: conda-forge
- NumPy: 2.2.6
- Matplotlib: 3.10.9
- OpenCV: 4.13.0
- Ultralytics: 8.4.56
- PyTorch: 2.12.0+cu126
- CUDA utilizada por PyTorch: 12.6
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU

En el equipo utilizado durante el trabajo, el ejecutable correspondiente era:

C:\conda_envs\tfg_yolo_cuda\python.exe

La ruta anterior es únicamente una referencia del entorno original y no es
necesaria en otras instalaciones.

El script principal asociado a este entorno es:

scripts/inference/inferir_comparativa_conf_010_015_020_025.py

La configuración de inferencia territorial de referencia fue:

- imgsz = 1280
- max_det = 3000
- IoU de NMS = 0,70
- device = 0
- conf = 0,10; 0,15; 0,20; 0,25

El modelo esperado por el script debe situarse en:

models/best.pt

## 2. Entorno de evaluación y visualización

Las fases de evaluación del detector, tratamiento tabular y generación de
figuras fueron verificadas también con una instalación independiente de Python.

Configuración comprobada:

- Python: 3.12.10
- NumPy: 2.4.2
- pandas: 3.0.2
- Matplotlib: 3.10.8
- OpenCV: 4.11.0
- Ultralytics: 8.4.46
- PyTorch: 2.11.0+cpu
- CUDA disponible en este entorno: no

Este entorno se utilizó satisfactoriamente para reproducir la evaluación
TP/FP/FN y las visualizaciones incluidas en el repositorio.

Entre los scripts asociados se encuentran:

scripts/evaluation/evaluar_tp_fp_fn.py
scripts/evaluation/analizar_resultados_tp_fp_fn.py
scripts/visualization/generar_visualizaciones_biomasa_por_recorte_AJUSTADO.py
scripts/visualization/pintar-cajas-amarillo-comparativa-conf.py

La evaluación reproducida desde el repositorio genera resultados
científicamente idénticos a los resultados congelados publicados al excluir
los campos que contienen rutas locales dependientes de la máquina.

## 3. Entorno geoespacial QGIS / GDAL

Los scripts que utilizan `osgeo` requieren un entorno con GDAL correctamente
configurado.

En el equipo empleado se verificó el entorno incluido con QGIS:

- QGIS: 3.44.9
- Python de QGIS: 3.12.13
- NumPy: 2.4.3
- GDAL: 3.12.3 "Chicoutimi"

La importación:

from osgeo import gdal, osr

se comprobó correctamente desde el entorno Python inicializado por QGIS.

En Windows, el entorno puede ejecutarse mediante el lanzador de QGIS. En la
instalación utilizada durante el trabajo:

C:\Program Files\QGIS 3.44.9\bin\python-qgis-ltr.bat

Ejemplo:

"C:\Program Files\QGIS 3.44.9\bin\python-qgis-ltr.bat" script.py

Los principales scripts asociados a este entorno son:

scripts/georeferencing/exportar_detecciones_lidar_comparativa_conf.py
scripts/biomass/calcular_biomasa_por_recorte_comparativa_conf.py

El Python general del sistema y el entorno CUDA no disponían de `osgeo`, por lo
que no deben utilizarse para estos scripts salvo que GDAL se instale y configure
explícitamente.

## 4. Dependencias por fase

| Fase | Dependencias principales |
| --- | --- |
| Evaluación del detector | Python, NumPy |
| Análisis de resultados | Python, pandas |
| Inferencia YOLO | Ultralytics, PyTorch, NumPy, OpenCV |
| Georreferenciación | NumPy, GDAL/osgeo |
| Biomasa y análisis espacial | NumPy, GDAL/osgeo |
| Visualización de resultados | pandas, NumPy, Matplotlib |
| Representación de cajas | OpenCV |

## 5. Consideraciones de reproducibilidad

Las versiones anteriores corresponden a entornos que fueron inspeccionados y
verificados durante la preparación final del repositorio. No se pretende afirmar
que todas las versiones sean requisitos mínimos estrictos.

La instalación de GDAL en Windows puede depender de la distribución empleada.
Por este motivo, para la fase geoespacial se recomienda utilizar el entorno
Python proporcionado por QGIS o una distribución geoespacial equivalente en la
que `from osgeo import gdal, osr` funcione correctamente.

No se proporciona un único archivo `requirements.txt` que mezcle todas las
dependencias porque el flujo histórico y reproducido del trabajo utilizó
entornos diferenciados para las fases de GPU, análisis y procesamiento
geoespacial.

## 6. Comprobaciones rápidas

Entorno CUDA:

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

Ultralytics:

python -c "import ultralytics; print(ultralytics.__version__)"

GDAL desde el entorno de QGIS:

python -c "from osgeo import gdal; print(gdal.VersionInfo('--version'))"

Las rutas concretas de los ejecutables dependen de cada instalación y no forman
parte de la configuración portable del repositorio.
