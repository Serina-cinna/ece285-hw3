"""Download and extract the assigned Histopathology collection."""

import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path


URL = "https://www.kaggle.com/api/v1/datasets/download/rushilprajapati/data-final?datasetVersionNumber=1"
EXPECTED_IMAGES = 5179


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        if offset and response.status != 206:
            offset = 0
            mode = "wb"
        else:
            mode = "ab" if offset else "wb"
        total = response.headers.get("Content-Range", "").split("/")[-1]
        total = int(total) if total.isdigit() else None
        done, last = offset, time.time()
        with partial.open(mode) as stream:
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
                done += len(block)
                if time.time() - last > 10:
                    print(f"Downloaded {done}/{total or '?'} bytes", flush=True)
                    last = time.time()
    if total and partial.stat().st_size != total:
        raise RuntimeError(f"Incomplete download: {partial.stat().st_size}/{total} bytes")
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="directory for downloaded data")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the full Kaggle archive after extracting the assigned subset")
    args = parser.parse_args()
    root = Path(args.data_root)
    archive = root / "data-final.zip"
    images = root / "Histopathology"
    if not archive.exists():
        download(URL, archive)
    images.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.startswith("data/Histopathology/") and not name.endswith("/")]
        if len(members) != EXPECTED_IMAGES:
            raise RuntimeError(f"Expected {EXPECTED_IMAGES} Histopathology images, found {len(members)}")
        for i, name in enumerate(members, 1):
            target = images / Path(name).name
            if not target.exists():
                with bundle.open(name) as source, target.open("wb") as destination:
                    while block := source.read(1024 * 1024):
                        destination.write(block)
            if i % 1000 == 0:
                print(f"Extracted {i}/{len(members)} images", flush=True)
    actual = len([p for p in images.iterdir() if p.is_file()])
    if actual != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} extracted images, found {actual}")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    manifest = {
        "dataset": "rushilprajapati/data-final",
        "version": 1,
        "source_url": URL,
        "subset": "data/Histopathology",
        "image_count": actual,
        "archive_sha256": digest.hexdigest(),
    }
    (root / "dataset_source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.keep_archive:
        archive.unlink()
    print(f"Ready: {actual} images in {images}", flush=True)


if __name__ == "__main__":
    main()
