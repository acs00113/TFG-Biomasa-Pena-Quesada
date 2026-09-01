# Datos geoespaciales externos

Este directorio está reservado para los datos geoespaciales externos empleados
en las fases territoriales del Trabajo Fin de Grado.

Los archivos ráster, las teselas PNOA y los datos LiDAR no se incluyen en el
historial Git debido a su tamaño. Este documento describe su procedencia,
características y ubicación esperada para reproducir el procesamiento.

## 1. Estructura esperada

Los scripts del repositorio esperan la siguiente organización:

data/external/
├── README.md
├── pnoa_tiles/
│   ├── png/
│   └── geotiff/
└── lidar/
    └── Altura_Árboles_Biomasa.tif

Las imágenes PNG y los GeoTIFF de una misma tesela deben conservar el mismo
nombre base para poder establecer su correspondencia durante la
georreferenciación.

## 2. Ortofotografía PNOA

La fuente óptica utilizada en el trabajo corresponde a la ortofotografía del
Plan Nacional de Ortofotografía Aérea (PNOA) distribuida por el IGN/CNIG.

Archivo original utilizado:

PNOA_MA_OF_ETRS89_HU30_h25_0949_2.tif

Características registradas durante el desarrollo del proyecto:

- Sistema de referencia: ETRS89 / UTM zona 30N
- EPSG: 25830
- Resolución espacial: 0,25 m/píxel
- Dimensiones del archivo original: 59 392 × 37 632 píxeles
- Origen aproximado: X = 498110 m; Y = 4187390 m
- Extensión aproximada:
  - X: 498110 – 512958 m
  - Y: 4177982 – 4187390 m

La ortofotografía original no es necesaria para ejecutar los scripts de
inferencia incluidos en este repositorio si se dispone previamente de las
teselas territoriales descritas a continuación.

## 3. Recorte territorial y teselado PNOA

El área experimental utilizada corresponde a la extensión común PNOA–LiDAR:

- X: 508000 – 512000 m
- Y: 4179000 – 4183000 m
- Dimensiones espaciales: 4000 × 4000 m
- Superficie total: 16 km² (1600 ha)
- Sistema de referencia: EPSG:25830

El recorte PNOA mantiene la resolución de 0,25 m/píxel, por lo que su tamaño es
de aproximadamente 16 000 × 16 000 píxeles.

Para la inferencia territorial se generó una rejilla de teselas con:

- Tamaño nominal: 1920 × 1080 píxeles
- Solape: 192 píxeles
- Solape sobre el terreno: 48 m
- Columnas: 10
- Filas: 18
- Posiciones totales procesadas: 180

Las teselas situadas en los bordes pueden presentar dimensiones inferiores al
tamaño nominal.

Para cada posición existen dos representaciones:

1. PNG, utilizada como entrada del detector YOLO.
2. GeoTIFF, utilizada posteriormente para recuperar la georreferenciación.

Las teselas deben colocarse en:

data/external/pnoa_tiles/png/

y:

data/external/pnoa_tiles/geotiff/

La generación histórica de las teselas se realizó mediante herramientas
GDAL/OSGeo. El comando de referencia utilizado para el teselado fue equivalente
a:

python -m osgeo_utils.gdal_retile -v -of GTiff -ps 1920 1080 -overlap 192 -co TILED=YES -co COMPRESS=LZW -targetDir <DIRECTORIO_SALIDA> <PNOA_RECORTADO>

Posteriormente, cada GeoTIFF se exportó también a PNG para su uso con el
detector.

## 4. Datos LiDAR

La información altimétrica procede de la cobertura LiDAR-PNOA correspondiente a
Andalucía, con archivos identificados mediante el patrón:

PNOA_2024_AND_<X>-<Y>_H30_NPC01

Se conservaron archivos LAZ y COPC-LAZ para 14 posiciones espaciales distintas:

508-4180
508-4181
509-4180
509-4181
509-4182
509-4183
510-4180
510-4181
510-4182
510-4183
511-4180
511-4181
511-4182
511-4183

Estos archivos originales no son necesarios para ejecutar directamente los
scripts de georreferenciación y estimación de biomasa si se dispone del raster
de altura empleado en el trabajo.

## 5. Raster de altura utilizado

El producto altimétrico empleado directamente por los scripts es:

Altura_Árboles_Biomasa.tif

Debe situarse en:

data/external/lidar/Altura_Árboles_Biomasa.tif

Características verificadas:

- Sistema de referencia: ETRS89 / UTM zona 30N
- EPSG: 25830
- Resolución espacial: 1 m/píxel
- Dimensiones: 4000 × 4000 píxeles
- Extensión:
  - X: 508000 – 512000 m
  - Y: 4179000 – 4183000 m
- Tamaño del archivo utilizado: 64 049 218 bytes
- Tamaño aproximado: 61,08 MB

SHA-256:

177FB042B1CEBC19D2E9C7FBD7C251C1B80D2AB1AD00CEA35D8E8B88C221FD17

Puede comprobarse su integridad en Windows PowerShell mediante:

Get-FileHash ".\data\external\lidar\Altura_Árboles_Biomasa.tif" -Algorithm SHA256

El raster contiene alturas de vegetación y no valores de biomasa, pese a la
denominación histórica del archivo.

Durante el procesamiento se consideran válidos únicamente los valores de altura
comprendidos entre 2 y 40 m.

## 6. MDS y MDT

Durante el desarrollo del proyecto se dispuso también de los productos:

MDS_final.tif
MDT_final.tif

ambos con una resolución espacial de 1 m/píxel y extensión coincidente con el
área experimental.

Conceptualmente, el raster de altura de vegetación se obtiene mediante:

altura = MDS - MDT

La documentación y los productos intermedios conservados respaldan este flujo
general. Sin embargo, no se ha podido reconstruir de forma completa y
reproducible la totalidad de los parámetros empleados en la rasterización e
interpolación originales.

Por este motivo, el repositorio no presenta esa fase como una cadena de
procesamiento completamente automatizada y utiliza como entrada reproducible el
raster final de altura identificado mediante su SHA-256.

## 7. Datos incluidos y datos no incluidos

Los datos PNOA y LiDAR originales no se redistribuyen dentro del historial Git.

El usuario que desee reproducir las fases territoriales debe obtener los datos
correspondientes de las fuentes oficiales del IGN/CNIG y preparar las entradas
siguiendo la estructura descrita en este documento.

Los productos derivados de fuentes oficiales deben utilizarse y citarse de
acuerdo con las condiciones de uso y atribución establecidas por el organismo
proveedor.
