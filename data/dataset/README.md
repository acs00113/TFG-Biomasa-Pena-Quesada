# Conjunto de datos anotado

Este directorio contiene el conjunto de datos local utilizado para el ajuste fino
y la evaluación del detector de arbolado desarrollado en el Trabajo Fin de Grado.

## Composición

El conjunto contiene 19 imágenes RGB:

- 14 imágenes de entrenamiento.
- 5 imágenes de validación.

Todas las imágenes tienen una resolución de 1920 × 1080 píxeles y se utilizan
con una única clase:

`Tree` (clase 0)

Las etiquetas se almacenan en formato YOLO.

La estructura es:

data/dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml

Las cinco imágenes de validación contienen conjuntamente 1266 anotaciones de
referencia.

## Procedencia de las imágenes

Las imágenes corresponden a recortes/capturas preparados a partir de
ortofotografía aérea PNOA de la zona de estudio de Peña de Quesada (Jaén).

La fuente geográfica utilizada en el proyecto fue:

`PNOA_MA_OF_ETRS89_HU30_h25_0949_2.tif`

Producto: Ortofotos PNOA Máxima Actualidad.

Proveedor: Instituto Geográfico Nacional / Centro Nacional de Información
Geográfica (IGN/CNIG).

Los PNG empleados para entrenamiento y validación no conservan
georreferenciación interna. Fueron preparados específicamente para las fases de
anotación, entrenamiento y evaluación del detector.

La información PNOA y sus productos derivados están sujetos a las condiciones
de utilización establecidas por el IGN/CNIG, mediante una licencia de uso
compatible con CC-BY 4.0 que exige reconocer el origen y la propiedad de los
datos.

Fuente de las imágenes: Ortofotos PNOA Máxima Actualidad, IGN/CNIG.

## Anotaciones

Las anotaciones de copas arbóreas fueron realizadas manualmente durante el
desarrollo de este TFG mediante LabelImg.

Cada copa visualmente distinguible se delimitó mediante una caja rectangular y
se asignó a la única clase utilizada:

`0 Tree`

Las anotaciones se almacenan mediante el formato YOLO normalizado:

`class_id x_center y_center width height`

Las coordenadas están normalizadas respecto al ancho y alto de cada imagen.

## Partición de validación

Las imágenes utilizadas para validación son:

- `Alta-densidad_02.png`
- `Alta_densidad_objetos_04.png`
- `Baja_densidad_03.png`
- `Baja_densidad_05.png`
- `Densidad_media_07.png`

Estas cinco imágenes suman 1266 cajas de referencia y son las utilizadas por
`scripts/evaluation/evaluar_tp_fp_fn.py` para calcular TP, FP, FN, precisión,
exhaustividad y F1.

## Configuración

El archivo `data.yaml` utiliza rutas relativas:

`train: images/train`

`val: images/val`

y define una única clase:

`0: Tree`

De esta forma el conjunto de datos puede utilizarse desde una copia del
repositorio sin depender de rutas absolutas del equipo original.

## Distinción respecto al teselado territorial

Este conjunto de 19 imágenes no debe confundirse con las 180 posiciones del
teselado territorial utilizado posteriormente para realizar la inferencia sobre
las 1600 ha del área experimental.

Las imágenes de este directorio se emplean para entrenamiento y evaluación del
detector. Las teselas territoriales PNOA utilizadas para la inferencia completa
se documentan por separado en:

`data/external/README.md`
