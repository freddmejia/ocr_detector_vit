# TrOCR sobre IAM con Docker

Entorno para el [cuaderno de Niels Rogge](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb).
Incluye JupyterLab, Python 3.11, PyTorch 2.7.1 (CUDA 12.8), Transformers 4.44.2,
Datasets 2.21.0 y Accelerate. Las versiones de Transformers y Datasets conservan
las API que utiliza el cuaderno (`evaluation_strategy`, `load_metric` y
`processor.feature_extractor`). Las dependencias transitivas no están bloqueadas.

La imagen incluye además un segundo entorno para el detector de placas RF-DETR
(ver [Detector de placas con RF-DETR](#detector-de-placas-con-rf-detr)).

## Obtener los datos

El cuaderno original presupone que ya dispones de IAM en Google Drive; no lo
descarga. Aquí se incluye un descargador de la [copia pública Teklia/IAM-line en
Hugging Face](https://huggingface.co/datasets/Teklia/IAM-line). Es una versión
preparada de IAM con las imágenes de líneas redimensionadas a 128 píxeles de alto,
no los archivos originales a resolución completa.

Desde la carpeta del proyecto:

```bash
docker compose build
docker compose run --rm trocr python scripts/download_iam.py
docker compose up
```

La descarga de los tres archivos Parquet ocupa aproximadamente 266 MB; deja
espacio adicional para la caché y las imágenes exportadas. No requiere GPU.
El script guarda los datos en la carpeta local montada en el contenedor:

```text
data/IAM/
├── source.json        # procedencia, revisión, recuentos y estado de descarga
├── train.txt          # 6.482 ejemplos de entrenamiento
├── validation.txt     # 976 ejemplos de validación
├── test.txt           # 2.915 ejemplos de prueba
└── image/teklia/
    ├── train/
    ├── validation/
    └── test/
```

El script fija la revisión `fbdad97500ce54635c0d1ba306bf535cb40656cf`, verifica las
imágenes y exporta sus bytes sin volver a comprimirlos. Puedes repetir el comando
si se interrumpe: reutiliza las descargas en caché y los archivos ya exportados.
Los datos están excluidos de Git y de la imagen Docker.

El cuaderno detecta estos archivos automáticamente, entrena con `train.txt` y
valida con `validation.txt`. Mantiene `test.txt` separado para una evaluación
final; no mezcla el conjunto de prueba con el entrenamiento.

### Obtener los originales de IAM

Para descargar las imágenes originales, entra en la [página oficial de
descarga](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database),
regístrate siguiendo sus indicaciones y descarga **lines.tgz** (imágenes de
líneas) y **ascii.tgz** (anotaciones, incluido `lines.txt`). IAM indica uso para
investigación no comercial y solicita registro y cita del artículo. Consulta
esas condiciones también si utilizas una copia redistribuida.

El descargador de este proyecto trabaja con la versión de Teklia, no con esos
archivos oficiales. `lines.txt` y `words.txt` tienen un formato distinto y no
deben renombrarse directamente a `train.txt` o `gt_test.txt`.

### Datos propios o formato del tutorial original

Como alternativa, puedes colocar `data/IAM/gt_test.txt` e imágenes en
`data/IAM/image/`. Cada línea debe tener `nombre_imagen transcripción`, por ejemplo
`a01-000u-00.jpg A MOVE to stop Mr. Gaitskell`. Solo si faltan las particiones
anteriores, el cuaderno usa este archivo y lo divide 80/20 como el tutorial.

## Arrancar

Desde esta carpeta, con Docker Desktop usando contenedores Linux:

```bash
docker compose up --build
```

Abre la dirección `http://localhost:8888/lab?token=...` que aparece en los logs.
Jupyter genera un token al iniciar. Para consultar el enlace después:

```bash
docker compose exec trocr micromamba run -n trocr jupyter server list
```

Abre `notebooks/TrOCR_IAM.ipynb` y selecciona el kernel **Python (TrOCR)**.
La copia está adaptada a las rutas del contenedor, desactiva FP16 en CPU y guarda
checkpoints en `outputs/`. Mantiene los modelos del original:
`microsoft/trocr-base-handwritten` para el procesador y
`microsoft/trocr-base-stage1` para el modelo.
Las celdas de instalación se han retirado: las dependencias ya están en la imagen.
La configuración de generación también se copia a `model.generation_config`.

El primer build y la primera carga del modelo requieren Internet y varios GB de
espacio. Los cuadernos, datos y resultados persisten en esta carpeta; los modelos
y métricas descargados persisten en el volumen `huggingface-cache`.

## Entrenar con GPU NVIDIA

Necesitas una GPU NVIDIA y un controlador compatible con CUDA 12.8. En Windows,
usa el backend WSL2 de Docker Desktop con acceso a la GPU. En Linux, configura
NVIDIA Container Toolkit para Docker.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Comprueba el acceso desde otra terminal:

```bash
docker compose exec trocr micromamba run -n trocr python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); x = torch.ones(16, device='cuda'); print((x + x).sum().item()); torch.cuda.synchronize()"
```

Esta prueba ejecuta una operación real en la GPU: `torch.cuda.is_available()`
por sí solo no garantiza que la versión instalada soporte su arquitectura.
Debe imprimir `2.7.1+cu128`, `12.8`, el nombre de tu GPU y `32.0`.

Si construiste la imagen anterior con PyTorch 2.6 / CUDA 12.4, reconstruye y
recrea el servicio (guarda antes el cuaderno abierto):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build --force-recreate
```

Después vuelve a abrir el cuaderno y ejecútalo desde el principio con un kernel
nuevo. Las RTX 50, incluida la RTX 5060 Ti (`sm_120`), requieren una distribución
de PyTorch con soporte Blackwell, como la CUDA 12.8 fijada aquí. La combinación
anterior podía detectar la GPU pero fallaba con `no kernel image is available`.

La configuración básica permite ejecutar en CPU, aunque el entrenamiento es
mucho más lento. Si falta memoria de GPU, reduce los batch sizes de 8 a 1 o 2
en `Seq2SeqTrainingArguments`.

## Detector de placas con RF-DETR

`notebooks/RFDETR_License_Plates.ipynb` adapta la guía
[Object detection](https://huggingface.co/docs/transformers/tasks/object_detection)
de Transformers: ajusta `Roboflow/rf-detr-medium` con el dataset
[justjuu/license-plate-detection](https://huggingface.co/datasets/justjuu/license-plate-detection)
(una sola clase, `license_plate`; 6.176 imágenes de entrenamiento, 1.765 de
validación y 882 de prueba). Sirve para localizar la placa antes de pasar el
recorte al OCR.

RF-DETR requiere Transformers 5, incompatible con la versión que necesita el
cuaderno de TrOCR, así que tiene su propio entorno Conda, `detr`, definido en
`env-detr.yml` (PyTorch 2.7.1 CUDA 12.8, Transformers 5.17.0, Datasets 5.0.1,
timm, Albumentations, torchmetrics, pycocotools y trackio). En JupyterLab, abre
el cuaderno y selecciona el kernel **Python (RF-DETR)**. Este entorno añade unos
GB a la imagen; reconstrúyela con `docker compose up --build`.

El dataset se descarga en la primera ejecución a la caché de Hugging Face (unos
240 MB, revisión fijada). El cuaderno entrena con `train`, elige el mejor
checkpoint con `validation` y evalúa al final con `test`. Los checkpoints y el
modelo final se guardan en `outputs/rf_detr_license_plates/`. El cuaderno
muestra las métricas por época (mAP, mAR); además trackio las registra en local,
dentro del volumen `huggingface-cache`, sin crear ningún Space.

Si falta memoria de GPU, reduce `per_device_train_batch_size` de 8 a 4 o 2.
Para subir el modelo al Hub, pon `PUSH_TO_HUB = True` en la primera celda.

La última celda exporta el detector a TFLite (LiteRT) con `litert-torch`, sin
TensorFlow: `outputs/rf_detr_license_plates/rf_detr_license_plates.tflite`
(~125 MB, float32). Requiere haber ejecutado antes las celdas de inferencia.
Entrada: `pixel_values` float32 `[1, 3, 576, 576]` (NCHW, normalizado con la
media y desviación de ImageNet); salidas: `logits` `[1, 300, 1]` (confianza =
sigmoide) y `pred_boxes` `[1, 300, 4]` (`cx, cy, w, h` normalizados). La celda
comprueba con el intérprete de LiteRT que da las mismas detecciones que PyTorch.
`litert-torch` solo tiene versión para Linux, así que la exportación funciona
dentro del contenedor, no en Python de Windows.

## OCR de placas con UC3M-LP

`notebooks/TrOCR_Placas.ipynb` (kernel **Python (TrOCR)**) adapta el cuaderno de
IAM para ajustar TrOCR (`microsoft/trocr-base-stage1`) con placas españolas del
dataset UC3M-LP. Coloca el dataset en `data/ocr_vehicles/UC3M-LP/` con su
estructura original: `train/` y `test/` (cada imagen con su JSON) y `train.txt`
y `test.txt`.

La primera ejecución recorta todas las placas (unas 2.500, a partir de 4,4 GB de
fotos) en `data/ocr_vehicles/UC3M-LP_ocr/`, con margen extra alrededor de cada
placa, y escribe `train.csv`, `validation.csv` y `test.csv` con el texto y la
posición de la placa en el recorte. La validación se separa de `train`
agrupando por placa. UC3M-LP tapa dos caracteres de cada placa con un bloque
gris; el texto objetivo los marca con `*` (`074*C*V`). El campo `imagePath` de
los JSON está desplazado una posición, así que el cuaderno toma la imagen con el
mismo nombre que el JSON. La carpeta `UC3M-LP_crops/` de la versión anterior ya
no se usa y puedes borrarla.

En entrenamiento, cada placa se aumenta al leerla: margen aleatorio, rotación
leve, brillo, contraste y color, desenfoque, compresión JPEG y reducción a baja
resolución (60–200 px de ancho, como las placas del detector). Validación y test
usan el mismo recorte que el pipeline.

La entrada al modelo es de 192×384 píxeles (`IMAGE_SIZE`) en lugar de los
384×384 de TrOCR: una placa es ancha, así que se conserva la resolución
horizontal con la mitad de parches y el codificador tarda la mitad. Los
*embeddings* de posición del codificador se interpolan al nuevo tamaño al cargar
el modelo, y el modelo guardado ya lo incluye, así que el pipeline no cambia.

Entrena con batch 16 hasta 40 épocas, evalúa cada época (CER y porcentaje de
placas exactas) y se detiene si el CER no mejora en 5 épocas. Guarda los
checkpoints en `outputs/trocr_placas/checkpoints` y el mejor modelo en
`outputs/trocr_placas/final`. Al final evalúa con `test` a resolución completa y
con las placas reducidas a 100 px. Si falta memoria de GPU, usa batch 8 con
`gradient_accumulation_steps=2`.

La última celda exporta el OCR a TFLite (LiteRT) con
`scripts/exportar_ocr_tflite.py`, que se ejecuta con el Python del entorno
`detr` (`litert-torch` no es compatible con el entorno de TrOCR). TrOCR genera
el texto token a token, así que el `.tflite` tiene dos firmas: `encoder`
(`pixel_values` → características) y `decoder` (`input_ids` int32 `[1, 16]` +
características → `logits`); el bucle de lectura (voraz, sin caché) va fuera del
modelo. En `outputs/<modelo>/tflite/` quedan `trocr_placas.tflite` (float32,
~1,5 GB), `trocr_placas_int8.tflite` (pesos int8, ~400 MB), `vocabulario.json` y
`config_tflite.json`. La celda compara las lecturas de PyTorch y de los dos
`.tflite` en placas de test. El script también se puede usar directamente:

```bash
docker compose exec trocr micromamba run -n detr python scripts/exportar_ocr_tflite.py \
    outputs/trocr_placas_rellenas_colombia/final
```

### Placas sin anonimizar (rellenadas)

`notebooks/Rellenar_UC3M-LP.ipynb` (kernel **Python (TrOCR)**) sustituye los
bloques grises de UC3M-LP por caracteres reales: localiza cada bloque en el
hueco entre sus caracteres vecinos (zona de color uniforme distinta del fondo) y
pinta una cifra o letra aleatoria tomada de un banco con los caracteres visibles
de todas las placas, adaptada a la altura, colores, nitidez e inclinación de la
placa. La etiqueta recibe ese carácter, así que el texto ya no tiene `*`. Solo
usa las placas de una fila con formato actual (4 cifras y 3 letras) y descarta
aquellas en las que no localiza los bloques con seguridad. Genera
`data/ocr_vehicles/UC3M-LP_ocr_rellenas/` con el mismo formato y las mismas
placas de validación que `TrOCR_Placas.ipynb`.

Con `USAR_RELLENAS = True` (valor por defecto), `TrOCR_Placas.ipynb` entrena con
ese dataset y guarda el modelo en `outputs/trocr_placas_rellenas/`. El pipeline
usa ese modelo si existe y, si no, el de `outputs/trocr_placas/`.

### Placas colombianas

Con `USAR_COLOMBIANAS = True` (valor por defecto), `TrOCR_Placas.ipynb` añade las
placas de `data/ocr_vehicles/new_plates/plates-ocr-train` (repositorio
[jdbravo/plates-ocr-train](https://gitlab.com/jdbravo/plates-ocr-train): recortes
de placas colombianas con `train_annotations.csv` y `valid_annotations.csv`).
Pasa las etiquetas a mayúsculas, descarta las que no siguen el formato
colombiano (`AAA000` en coches, `AAA00A` en motos), deja en entrenamiento las
placas que el repositorio repite entre `train` y `valid` y reparte el resto de
`valid` entre validación y test. Cada placa colombiana aparece dos veces por
época (`COLOMBIA_REPEAT`). El modelo se guarda con el sufijo `_colombia`
(`outputs/trocr_placas_rellenas_colombia`), el pipeline lo usa si existe, y la
evaluación final da las métricas por fuente.

## Pipeline completo: placa + OCR

`notebooks/Pipeline_Placas_OCR.ipynb` (kernel **Python (RF-DETR)**) une los dos
modelos entrenados: lee las imágenes de `data/vehicles`, las normaliza y
redimensiona para el detector, detecta las placas con
`outputs/rf_detr_license_plates/final`, recorta cada placa y la lee con el TrOCR de placas
(`outputs/trocr_placas/final`). Guarda en `outputs/pipeline/` cada imagen anotada con la caja
y el texto, y un `lecturas.csv` con todas las lecturas. La función `leer_placas`
aplica todo el pipeline a una imagen nueva.

Parámetros en la primera celda: `DET_THRESHOLD` (confianza mínima, 0.4),
`CROP_PADDING` (margen del recorte) y `MIN_OCR_WIDTH` (60 px: las placas más
estrechas se dibujan, pero no se leen, porque a esa resolución el OCR solo
devuelve ruido).

El OCR se entrenó solo con placas españolas: con placas de otros países tenderá
a leer con el formato español.

Al final del cuaderno, una celda compara TrOCR con
[fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr)
(`cct-s-v2-global-model`, preentrenado con placas de más de 65 países, sin
ajustar) sobre los mismos recortes, y guarda la tabla y las imágenes anotadas en
`outputs/pipeline/fast_plate_ocr/`. Usa el paquete `fast-plate-ocr[onnx]` del
entorno `detr`; en un contenedor ya levantado se puede instalar sin reconstruir:

```bash
docker compose exec trocr micromamba run -n detr pip install "fast-plate-ocr[onnx]==1.1.0"
```

## Pipeline con los modelos TFLite

`notebooks/Pipeline_TFLite.ipynb` (kernel **Python (RF-DETR)**) prueba los dos
modelos exportados como en una aplicación: solo NumPy, PIL y el intérprete de
LiteRT, sin PyTorch. Carga los `.tflite` y muestra sus firmas, ejecuta el
pipeline completo (detección, recorte y lectura token a token) sobre
`data/vehicles` con tiempos por modelo, mide el OCR con las placas colombianas de
`valid` que no están en `train` y compara detecciones y lecturas con los modelos
originales de PyTorch. Usa el OCR con pesos int8 (`OCR_VARIANTE`) y guarda las
imágenes anotadas y las lecturas en `outputs/pipeline_tflite/`.

`docs/PIPELINE_TFLITE_KMP.md` describe cómo integrar los dos `.tflite` en una app
Kotlin Multiplatform: ficheros a empaquetar (con tamaños y SHA-256), tensores de
entrada y salida, preprocesado y postprocesado exactos, bucle de lectura del OCR,
formato del resultado y pruebas con los valores esperados.

## Parar y configurar

```bash
docker compose down
```

Puedes cambiar el puerto creando un archivo `.env` con `JUPYTER_PORT=8889`.
`env.yml` define las dependencias de Conda/Python; `.env` configura Compose.
El puerto de Jupyter solo se publica en la interfaz local.

### Límite de recursos

Para no saturar ni calentar el equipo, el contenedor usa como máximo ~60 % del
anfitrión (i7-14700KF con 28 hilos y 64 GB): **16 CPU y 38 GB de RAM**, sin
swap adicional. Los hilos de PyTorch, NumPy y OpenCV se ajustan al mismo número
de CPU. Para cambiarlo, añade a `.env` (CPU en número entero):

```bash
CONTAINER_CPUS=12
CONTAINER_MEMORY=24g
```

Los límites se aplican al crear el contenedor, así que tras cambiarlos hay que
recrearlo (`docker compose ... up -d --force-recreate`). Para comprobarlos
mientras entrena: `docker stats`.

Docker Desktop ejecuta los contenedores dentro de una máquina virtual WSL2 que,
por defecto, puede usar todas las CPU y el 50 % de la RAM. Si quieres limitar
también esa máquina (por ejemplo, para otros contenedores), crea
`%USERPROFILE%\.wslconfig` con este contenido y ejecuta `wsl --shutdown`:

```ini
[wsl2]
processors=16
memory=38GB
```

Docker no puede limitar la GPU: el límite se aplica en Windows y afecta a todo el
equipo. `scripts/limitar_gpu.ps1` fija la frecuencia máxima de la GPU al 70 % de
la de fábrica (2163 de 3090 MHz en la RTX 5060 Ti), que es lo que más reduce
consumo y temperatura, y su consumo máximo al mínimo que admite (150 W de
180 W). Pide permisos de administrador y los ajustes se pierden al reiniciar
Windows, así que hay que ejecutarlo antes de cada sesión de entrenamiento:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\limitar_gpu.ps1                 # 70 %
powershell -ExecutionPolicy Bypass -File scripts\limitar_gpu.ps1 -Porcentaje 60
powershell -ExecutionPolicy Bypass -File scripts\limitar_gpu.ps1 -Quitar         # valores de fábrica
```

Para vigilarla mientras entrena:
`nvidia-smi --query-gpu=clocks.gr,power.draw,temperature.gpu --format=csv -l 5`.

Para usar las mismas dependencias fuera de Docker (Linux x86_64):

```bash
conda env create -f env.yml
conda activate trocr
jupyter lab
```

La copia del cuaderno usa las carpetas `data/IAM` y `outputs` del proyecto como
alternativa cuando no existen las variables de entorno del contenedor.

Referencias: [PyTorch 2.7.1 y CUDA](https://pytorch.org/get-started/previous-versions/#v271),
[soporte Blackwell](https://pytorch.org/blog/pytorch-2-7/),
[métricas en Datasets 2.21](https://huggingface.co/docs/datasets/v2.21.0/how_to_metrics),
[GPU en Compose](https://docs.docker.com/compose/how-tos/gpu-support/).
