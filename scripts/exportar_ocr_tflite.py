"""Exporta el OCR TrOCR ajustado (VisionEncoderDecoderModel) a TFLite (LiteRT).

TrOCR genera el texto token a token, así que el .tflite tiene dos firmas:

* ``encoder``: ``pixel_values`` float32 [1, 3, alto, ancho] -> características de la imagen.
* ``decoder``: ``input_ids`` int32 [1, max_length] + características -> ``logits`` [1, max_length, vocabulario].

El bucle de generación va fuera del modelo: se empieza con ``[decoder_start, pad, pad, ...]`` y, en cada paso ``t``,
el siguiente token es el argmax de ``logits[0, t]``, hasta el token de fin. Con longitud fija no hace falta caché.

Se ejecuta con el entorno ``detr`` (litert-torch no es compatible con el entorno de TrOCR):

    /opt/conda/envs/detr/bin/python scripts/exportar_ocr_tflite.py outputs/trocr_placas_rellenas_colombia/final \\
        --muestras muestras.csv

Genera en la carpeta de salida el modelo float32, una versión con pesos int8 (unas 4 veces más pequeña), el
vocabulario (``vocabulario.json``: id -> texto) y ``config_tflite.json`` con el preprocesado y los tokens especiales.
Si se pasa ``--muestras`` (CSV con columnas ``path`` y ``text``), compara las lecturas de PyTorch y de los .tflite.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def cargar_modelo(carpeta):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(carpeta)
    model = VisionEncoderDecoderModel.from_pretrained(carpeta).eval()
    # Guardado con Transformers 4.44: con Transformers 5 la tabla de posiciones sinusoidales queda sin datos
    posiciones = model.decoder.model.decoder.embed_positions
    if hasattr(posiciones, "get_embedding"):
        posiciones.weights = posiciones.get_embedding(
            model.config.decoder.max_position_embeddings + posiciones.padding_idx + 1,
            posiciones.embedding_dim,
            posiciones.padding_idx,
        )
    # Pesos en memoria propia: el conversor ignora el desplazamiento de los tensores que son vistas de otro
    with torch.no_grad():
        for parametro in model.parameters():
            parametro.data = parametro.data.clone()
    return processor, model


class Codificador(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        estados = self.model.encoder(pixel_values=pixel_values).last_hidden_state
        # misma proyección que VisionEncoderDecoderModel.forward cuando los tamaños no coinciden
        cfg = self.model.config
        if (cfg.encoder.hidden_size != cfg.decoder.hidden_size
                and getattr(cfg.decoder, "cross_attention_hidden_size", None) is None):
            estados = self.model.enc_to_dec_proj(estados)
        return estados


class Decodificador(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, encoder_hidden_states):
        salida = self.model.decoder(input_ids=input_ids.long(), encoder_hidden_states=encoder_hidden_states,
                                    use_cache=False)
        return salida.logits


def leer_torch_voraz(codificador, decodificador, pixel_values, tokens):
    # mismo bucle que se usará con el .tflite, en PyTorch
    with torch.no_grad():
        estados = codificador(pixel_values)
        ids = torch.full((1, tokens["max_length"]), tokens["pad"], dtype=torch.int32)
        ids[0, 0] = tokens["start"]
        for t in range(tokens["max_length"] - 1):
            siguiente = int(decodificador(ids, estados)[0, t].argmax())
            if siguiente == tokens["eos"]:
                break
            ids[0, t + 1] = siguiente
    return ids[0].tolist()


def lector_tflite(ruta, tokens):
    from ai_edge_litert.interpreter import Interpreter

    interprete = Interpreter(model_path=str(ruta))
    codificar = interprete.get_signature_runner("encoder")
    decodificar = interprete.get_signature_runner("decoder")

    def leer(pixel_values):
        estados = next(iter(codificar(args_0=pixel_values).values()))
        ids = np.full((1, tokens["max_length"]), tokens["pad"], dtype=np.int32)
        ids[0, 0] = tokens["start"]
        for t in range(tokens["max_length"] - 1):
            logits = next(iter(decodificar(args_0=ids, args_1=estados).values()))
            siguiente = int(logits[0, t].argmax())
            if siguiente == tokens["eos"]:
                break
            ids[0, t + 1] = siguiente
        return ids[0].tolist()

    return leer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modelo", type=Path, help="carpeta del modelo TrOCR guardado (…/final)")
    parser.add_argument("--salida", type=Path, help="carpeta de salida (por defecto <modelo>/../tflite)")
    parser.add_argument("--muestras", type=Path, help="CSV con columnas path,text para comparar las lecturas")
    parser.add_argument("--sin-int8", action="store_true", help="no generar la versión con pesos int8")
    args = parser.parse_args()

    import litert_torch

    salida = args.salida or args.modelo.parent / "tflite"
    salida.mkdir(parents=True, exist_ok=True)
    processor, model = cargar_modelo(args.modelo)
    size = processor.image_processor.size
    alto, ancho = size["height"], size["width"]
    gen = model.generation_config
    tokens = {"start": gen.decoder_start_token_id, "pad": gen.pad_token_id, "eos": gen.eos_token_id,
              "max_length": gen.max_length}
    print(f"Modelo {args.modelo}: entrada {alto}x{ancho}, max_length {tokens['max_length']}", flush=True)

    codificador, decodificador = Codificador(model).eval(), Decodificador(model).eval()
    pixel_values = torch.zeros(1, 3, alto, ancho)
    ids = torch.full((1, tokens["max_length"]), tokens["pad"], dtype=torch.int32)
    ids[0, 0] = tokens["start"]
    with torch.no_grad():
        estados = codificador(pixel_values)

    inicio = time.time()
    convertido = (litert_torch.signature("encoder", codificador, (pixel_values,))
                  .signature("decoder", decodificador, (ids, estados))
                  .convert())
    ruta_float = salida / "trocr_placas.tflite"
    convertido.export(str(ruta_float))
    modelos = {"float32": ruta_float}
    print(f"Convertido en {time.time() - inicio:.0f}s: {ruta_float} ({ruta_float.stat().st_size / 1e6:.0f} MB)",
          flush=True)

    if not args.sin_int8:
        from ai_edge_quantizer import quantizer, recipe

        ruta_int8 = salida / "trocr_placas_int8.tflite"
        cuantizador = quantizer.Quantizer(str(ruta_float), recipe.weight_only_wi8_afp32())
        cuantizador.quantize().export_model(str(ruta_int8), overwrite=True)
        modelos["int8"] = ruta_int8
        print(f"Pesos int8: {ruta_int8} ({ruta_int8.stat().st_size / 1e6:.0f} MB)", flush=True)

    tokenizer = processor.tokenizer
    vocabulario = {i: tokenizer.decode([i]) for i in range(len(tokenizer))}
    (salida / "vocabulario.json").write_text(json.dumps(vocabulario, ensure_ascii=False), encoding="utf-8")
    config = {
        "entrada": {"alto": alto, "ancho": ancho, "formato": "NCHW float32 RGB",
                    "reescalado": processor.image_processor.rescale_factor,
                    "media": list(processor.image_processor.image_mean),
                    "desviacion": list(processor.image_processor.image_std)},
        "tokens": tokens,
        "firmas": {"encoder": "args_0=pixel_values -> estados",
                   "decoder": "args_0=input_ids int32 [1, max_length], args_1=estados -> logits"},
    }
    (salida / "config_tflite.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Vocabulario y configuración en", salida, flush=True)

    if args.muestras:
        with open(args.muestras, encoding="utf-8") as f:
            muestras = list(csv.DictReader(f))
        lectores = {nombre: lector_tflite(ruta, tokens) for nombre, ruta in modelos.items()}
        aciertos = {"PyTorch beam search": 0, "PyTorch voraz": 0, **{f"TFLite {n}": 0 for n in lectores}}
        iguales = {n: 0 for n in lectores}
        tiempos = {n: 0.0 for n in lectores}
        for fila in muestras:
            imagen = Image.open(fila["path"]).convert("RGB")
            px = processor(images=imagen, return_tensors="pt").pixel_values
            with torch.no_grad():
                beam = processor.batch_decode(model.generate(px), skip_special_tokens=True)[0].strip()
            voraz_ids = leer_torch_voraz(codificador, decodificador, px, tokens)
            voraz = tokenizer.decode(voraz_ids, skip_special_tokens=True).strip()
            aciertos["PyTorch beam search"] += beam == fila["text"]
            aciertos["PyTorch voraz"] += voraz == fila["text"]
            for nombre, leer in lectores.items():
                inicio = time.time()
                lectura_ids = leer(px.numpy())
                tiempos[nombre] += time.time() - inicio
                lectura = tokenizer.decode(lectura_ids, skip_special_tokens=True).strip()
                aciertos[f"TFLite {nombre}"] += lectura == fila["text"]
                iguales[nombre] += lectura_ids == voraz_ids
        n = len(muestras)
        print(f"\nPlacas leídas sin errores ({n} muestras):")
        for nombre, total in aciertos.items():
            print(f"  {nombre:<22} {total / n:6.1%}")
        for nombre in lectores:
            print(f"  TFLite {nombre}: misma lectura que PyTorch voraz en {iguales[nombre]}/{n}; "
                  f"{tiempos[nombre] / n * 1000:.0f} ms por placa en CPU")


if __name__ == "__main__":
    main()
