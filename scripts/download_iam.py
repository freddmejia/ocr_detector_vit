"""Descarga IAM-line de Teklia y exporta imágenes y transcripciones para TrOCR."""

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm.auto import tqdm


REPO_ID = "Teklia/IAM-line"
REVISION = "fbdad97500ce54635c0d1ba306bf535cb40656cf"
EXPECTED_COUNTS = {"train": 6482, "validation": 976, "test": 2915}


def download(output: Path, revision: str) -> None:
    # Resuelve main una sola vez; todos los ficheros usan la misma revisión.
    commit = HfApi().dataset_info(REPO_ID, revision=revision).sha
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "source.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("revision") != commit:
            raise ValueError("La carpeta contiene otra revisión; usa --output con otra carpeta.")
    elif any((output / f"{split}.txt").exists() for split in EXPECTED_COUNTS):
        raise FileExistsError("Ya existen anotaciones sin source.json; usa otra carpeta.")

    metadata = {
        "dataset": REPO_ID,
        "url": f"https://huggingface.co/datasets/{REPO_ID}",
        "revision": commit,
        "image_height": 128,
        "complete": False,
        "counts": {},
    }
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for split, expected in EXPECTED_COUNTS.items():
        print(f"Descargando {split} ({commit})...", flush=True)
        parquet = hf_hub_download(
            REPO_ID, f"data/{split}.parquet", repo_type="dataset", revision=commit
        )
        destination = output / "image" / "teklia" / split
        destination.mkdir(parents=True, exist_ok=True)
        rows = []
        source = pq.ParquetFile(parquet)
        if source.metadata.num_rows != expected:
            raise ValueError(f"{split}: se esperaban {expected} filas, se encontraron {source.metadata.num_rows}.")
        with tqdm(total=expected, desc=f"Exportando {split}", mininterval=5) as progress:
            for batch in source.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    content = row["image"]["bytes"]
                    with Image.open(io.BytesIO(content)) as image:
                        extension = {"JPEG": ".jpg", "PNG": ".png"}[image.format]
                        image.verify()
                    text = " ".join(row["text"].split())
                    if not text:
                        raise ValueError(f"Transcripción vacía en {split}, fila {len(rows)}.")
                    relative = Path("teklia") / split / f"{len(rows):06d}{extension}"
                    target = output / "image" / relative
                    if not target.exists() or target.read_bytes() != content:
                        target.write_bytes(content)
                    rows.append(f"{relative.as_posix()} {text}\n")
                    progress.update(1)
        temporary = output / f"{split}.txt.tmp"
        temporary.write_text("".join(rows), encoding="utf-8")
        temporary.replace(output / f"{split}.txt")
        metadata["counts"][split] = len(rows)
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    metadata["complete"] = True
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"IAM preparado en {output.resolve()}: {metadata['counts']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/IAM"))
    parser.add_argument("--revision", default=REVISION, help="Commit de Hugging Face (se guarda en source.json).")
    args = parser.parse_args()
    download(args.output, args.revision)
