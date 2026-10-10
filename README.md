# Detección y lectura de matrículas

El proyecto se centra en **RF-DETR para detectar matrículas** y
**FastPlateOCR para reconocer su texto**, con exportaciones para una aplicación
móvil. El flujo previsto es: foto -> detector -> recorte de la placa -> OCR -> texto.

## Cuadernos conservados

| Cuaderno | Función | Kernel |
|---|---|---|
| `notebooks/RFDETR_License_Plates.ipynb` | Ajuste fino del detector y exportación a TFLite | Python (RF-DETR) |
| `notebooks/FastPlateOCR_Placas.ipynb` | Preparación del dataset OCR, ajuste fino, evaluación y exportación a TFLite/ONNX | Python (RF-DETR) |
| `notebooks/Rellenar_UC3M-LP.ipynb` | Sustituir bloques anonimizados por caracteres sintéticos y generar sus etiquetas | Python (TrOCR) |
| `notebooks/Pipeline_Placas_OCR.ipynb` | Pipeline por etapas: RF-DETR en PyTorch y FastPlateOCR en ONNX | Python (RF-DETR) |
| `notebooks/Pipeline_TFLite.ipynb` | Pipeline RF-DETR + FastPlateOCR con LiteRT y comparación con PyTorch/ONNX | Python (RF-DETR) |

**Estado de la limpieza:** se han retirado los cuadernos de entrenamiento de
TrOCR, los datos IAM, los recortes antiguos de UC3M-LP, sus scripts y los modelos
y checkpoints de TrOCR. Los cinco cuadernos están en inglés, sin salidas guardadas
ni contadores de ejecución. Los pipelines reutilizan la detección, los recortes y
las visualizaciones anteriores, con la inferencia FastPlateOCR del cuaderno de
entrenamiento. Guardan resultados nuevos en `outputs/pipeline/fastplate/` y
`outputs/pipeline_tflite/fastplate/`, respectivamente.

`docs/PIPELINE_TFLITE_KMP.md` y los resultados de `outputs/pipeline/` y
`outputs/pipeline_tflite/` se conservan como referencia del pipeline anterior.
La guía todavía describe RF-DETR Medium + TrOCR; no es el contrato de integración
para Nano + FastPlateOCR.

## Configurar los cuadernos

Las rutas y los parámetros principales están en la primera celda de código.
Se mantienen los valores de entrenamiento y las transformaciones existentes:

- **RF-DETR:** modelo, revisión del dataset, épocas, batch de entrenamiento y
  evaluación, acumulación de gradientes, workers, learning rate, weight decay,
  precisión BF16, aumentos, checkpoints y exportaciones. `RESUME_FROM_CHECKPOINT`
  permite reanudar; `OVERWRITE_EXPORT=True` regenera el TFLite tras reentrenar en
  el mismo directorio. Se ha unificado la exportación duplicada.
- **FastPlateOCR:** directorios, tamaño de entrada, repetición de Colombia,
  épocas, paciencia, batch, workers, learning rate, weight decay y aumentos.
  `RUN_TRAINING=False` permite usar una ejecución existente elegida con
  `SELECTED_RUN_DIR`. `WEIGHTS_OVERRIDE` carga pesos iniciales; no restaura el
  estado del optimizador. `RUN_EXPORT` controla la exportación.
- **Rellenado:** rutas, margen, tamaño máximo, semilla, partición de validación,
  umbrales de localización y tamaño del banco de caracteres. La generación no
  depende de ejecutar la celda de ejemplos.
- **Pipelines:** modelos, imágenes, resultados, umbral de detección, margen de
  recorte y ancho mínimo para OCR. El pipeline TFLite permite elegir float32,
  float16 o int8 para el detector y activar/desactivar evaluaciones adicionales.

Los datasets preparados se reutilizan por defecto. Si cambias los ajustes de
generación o recorte, utiliza otro directorio de datos o activa
`REBUILD_DATASET`. Se conservan los nombres de carpetas y las columnas existentes
`fuente` y `categoria` para mantener la compatibilidad de las anotaciones.

## Datos que se mantienen

```text
data/
|-- ocr_vehicles/
|   |-- UC3M-LP/                # originales: imágenes, JSON y particiones
|   |-- UC3M-LP.zip             # archivo original del dataset
|   |-- UC3M-LP_ocr_rellenas/    # recortes españoles y etiquetas sintéticas
|   |-- new_plates/
|   |   |-- plates-ocr-train/   # placas colombianas y anotaciones
|   |-- fast_plate_ocr/         # dataset preparado para FastPlateOCR
|-- vehicles/                  # nueve imágenes para probar el pipeline
```

Los datos están excluidos de Git y de la imagen Docker. La carpeta del proyecto
se monta en el contenedor, por lo que las imágenes y los resultados persisten
en el disco del anfitrión.

### Preparación del OCR

1. Los originales españoles proceden de
   [UC3M-LP](https://github.com/ramajoballester/UC3M-LP). Deben estar en
   `data/ocr_vehicles/UC3M-LP/`, con `train.txt`, `test.txt` y las carpetas
   `train/` y `test/`, cada imagen junto a su JSON.
2. `Rellenar_UC3M-LP.ipynb` genera `UC3M-LP_ocr_rellenas/`. Los caracteres
   insertados son aleatorios y las etiquetas se actualizan para coincidir con
   ellos; no se recuperan los caracteres originales ocultos.
3. Las placas colombianas proceden de
   [jdbravo/plates-ocr-train](https://gitlab.com/jdbravo/plates-ocr-train), en
   `new_plates/plates-ocr-train/`, con `train_annotations.csv`,
   `valid_annotations.csv` y sus imágenes.
4. `FastPlateOCR_Placas.ipynb` prepara `fast_plate_ocr/`, recortando las españolas
   con un margen horizontal del 8 % y vertical del 15 %, y copiando las
   colombianas. Las anotaciones usan `image_path` y `plate_text`.

El dataset preparado contiene:

| Fuente | Entrenamiento, sin repetir | Validación | Prueba |
|---|---:|---:|---:|
| UC3M-LP rellenadas | 1.620 | 181 | 432 |
| Colombia | 798 | 48 | 47 |
| Total | 2.418 | 229 | 479 |

Las colombianas de entrenamiento se repiten dos veces: `train.csv` tiene
3.216 filas. `train_unicas.csv` permite inspeccionar los ejemplos sin esa
repetición. `dataset.csv` conserva la fuente y la partición de cada imagen.

### Datos del detector

`RFDETR_License_Plates.ipynb` descarga
[justjuu/license-plate-detection](https://huggingface.co/datasets/justjuu/license-plate-detection)
en la caché de Hugging Face, con revisión fijada en el cuaderno. Usa una clase,
`license_plate`, y las particiones de 6.176 imágenes de entrenamiento, 1.765 de
validación y 882 de prueba. Esta caché está en el volumen Docker
`huggingface-cache`, separado de `data/`.

## Arrancar JupyterLab

Docker Desktop debe usar contenedores Linux. Para aprovechar la GPU NVIDIA:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Sin GPU:

```bash
docker compose up --build
```

Abre el enlace `http://localhost:8888/lab?token=...` de los logs. Para consultarlo:

```bash
docker compose exec trocr micromamba run -n trocr jupyter server list
```

La configuración Docker conserva los nombres de servicio e imagen `trocr` y
`trocr-iam:local`. Incluye dos entornos Python 3.11:

- `detr`, definido en `env-detr.yml`: RF-DETR, FastPlateOCR, Keras y herramientas
  de exportación. PyTorch 2.7.1 + CUDA 12.8 y Transformers 5.17.0.
- `trocr`, definido en `env.yml`: sirve JupyterLab y sigue proporcionando el
  kernel que utiliza el cuaderno de rellenado.

La limpieza de datos no reconstruye la imagen ni cambia los entornos.

Para detener el servicio:

```bash
docker compose down
```

## Modelos y exportaciones

### RF-DETR

El cuaderno selecciona `Roboflow/rf-detr-nano`. Entrena cinco épocas, selecciona
el mejor checkpoint por mAP y guarda el modelo en
`outputs/rf_detr_license_plates/final/`. El procesador del Nano usa 384x384.

Las exportaciones Nano existentes están empaquetadas en:

```text
outputs/rf_detr_license_plates/tflite/rf-detr-nano/rf_detr_license_plates_fp16.zip
```

El ZIP contiene las tres variantes: float32 (~110 MB), float16 (~57 MB) e int8
(~31 MB). El archivo suelto `outputs/rf_detr_license_plates/rf_detr_license_plates.tflite`
corresponde a la exportación anterior de Medium, a 576x576. No deben mezclarse
sus configuraciones. El procesador de `final/` está actualmente dentro de
`preprocessor_config.zip`; ambos pipelines pueden leerlo directamente. El
pipeline TFLite también puede cargar la variante seleccionada desde el ZIP
Nano cuando no existe el archivo suelto. Comprueba que las dimensiones de entrada
coincidan con la configuración del procesador.

### FastPlateOCR

El modelo base es `cct_s_v2_global`. Se ajusta con Keras sobre PyTorch y se
exporta con TensorFlow en CPU. La lectura produce diez posiciones sobre el
alfabeto `0-9A-Z_`; `_` es relleno. La entrada es RGB float32 en formato NHWC,
con valores 0-255 y normalización dentro del modelo.

```text
outputs/fast_plate_ocr/
|-- base/
|-- trained/                    # ejecuciones, best.keras y métricas
|-- export/
    |-- plate_ocr.tflite         # ~1,60 MB
    |-- plate_ocr.onnx           # ~4,92 MB
    |-- plate_config.yaml
```

El cuaderno está configurado a **96x192**, pero la exportación existente y su
`plate_config.yaml` corresponden al entrenamiento a **64x128**. Para usar un
modelo exportado, hay que tomar el tamaño de su configuración, no del valor
actual del cuaderno.

## Recursos del equipo

Compose limita el servicio a 16 CPU y 38 GB de RAM, sin swap adicional. Se puede
ajustar mediante `.env`:

```dotenv
JUPYTER_PORT=8888
CONTAINER_CPUS=16
CONTAINER_MEMORY=38g
```

`scripts/limitar_gpu.ps1` permite limitar frecuencia y potencia de la GPU NVIDIA
en Windows. Requiere permisos de administrador y afecta a toda la tarjeta:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\limitar_gpu.ps1
powershell -ExecutionPolicy Bypass -File scripts\limitar_gpu.ps1 -Quitar
```
