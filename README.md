# TrOCR sobre IAM con Docker

Entorno para el [cuaderno de Niels Rogge](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb).
Incluye JupyterLab, Python 3.11, PyTorch 2.6.0 (CUDA 12.4), Transformers 4.44.2,
Datasets 2.21.0 y Accelerate. Las versiones de Transformers y Datasets conservan
las API que utiliza el cuaderno (`evaluation_strategy`, `load_metric` y
`processor.feature_extractor`). Las dependencias transitivas no están bloqueadas.

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
docker compose exec trocr jupyter server list
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

Necesitas una GPU NVIDIA y un controlador compatible con CUDA 12.4. En Windows,
usa el backend WSL2 de Docker Desktop con acceso a la GPU. En Linux, configura
NVIDIA Container Toolkit para Docker.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Comprueba el acceso desde otra terminal:

```bash
docker compose exec trocr python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

La configuración básica permite ejecutar en CPU, aunque el entrenamiento es
mucho más lento. Si falta memoria de GPU, reduce los batch sizes de 8 a 1 o 2
en `Seq2SeqTrainingArguments`.

## Parar y configurar

```bash
docker compose down
```

Puedes cambiar el puerto creando un archivo `.env` con `JUPYTER_PORT=8889`.
`env.yml` define las dependencias de Conda/Python; `.env` configura Compose.
El puerto de Jupyter solo se publica en la interfaz local.

Para usar las mismas dependencias fuera de Docker (Linux x86_64):

```bash
conda env create -f env.yml
conda activate trocr
jupyter lab
```

La copia del cuaderno usa las carpetas `data/IAM` y `outputs` del proyecto como
alternativa cuando no existen las variables de entorno del contenedor.

Referencias: [PyTorch 2.6.0 y CUDA](https://pytorch.org/get-started/previous-versions/#v260),
[métricas en Datasets 2.21](https://huggingface.co/docs/datasets/v2.21.0/how_to_metrics),
[GPU en Compose](https://docs.docker.com/compose/how-tos/gpu-support/).
