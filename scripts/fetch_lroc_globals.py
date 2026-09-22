"""One-time download of the public, no-login global rasters + ground-truth
shapefile this project builds on:

- LROC WAC Global Morphology Mosaic, 100m/px (USGS Astrogeology)
- GLD100 DEM, 100m/px (USGS Astrogeology)
- Thompson et al. 2017 lunar wrinkle ridge shapefile (LROC / im-ldi.com)

All three URLs were found by fetching the public Astropedia/LROC pages and
extracting their direct download links (see research/DECISION_LOG.md) --
not guessed or hardcoded from memory.

Downloads are resumable (HTTP Range) and streamed to avoid loading a
multi-GB file into memory at once.
"""
from __future__ import annotations

import os
import zipfile

import requests

DATA_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

SOURCES = {
    "wac_mosaic.tif": "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif",
    "gld100_dem.tif": "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_WAC_GLD100_DTM_79S79N_100m_v1.1.tif",
    "wrinkle_ridges_180.zip": "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/SHAPEFILE/WRINKLE_RIDGES/WRINKLE_RIDGES_180.ZIP",
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def download_resumable(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> None:
    """Stream-download url to dest_path, resuming via HTTP Range if a
    partial file already exists from a previous interrupted run."""
    headers = {"User-Agent": USER_AGENT}
    mode = "wb"
    existing_size = 0

    if os.path.exists(dest_path):
        existing_size = os.path.getsize(dest_path)
        head = requests.head(url, headers=headers, timeout=30, allow_redirects=True)
        remote_size = int(head.headers.get("Content-Length", -1))
        if remote_size != -1 and existing_size == remote_size:
            print(f"  already complete ({existing_size:,} bytes), skipping")
            return
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
            print(f"  resuming from byte {existing_size:,}")

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + existing_size
        downloaded = existing_size
        with open(dest_path, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {downloaded:,}/{total:,} bytes ({pct:.1f}%)", end="\r")
        print()


def main() -> None:
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    for filename, url in SOURCES.items():
        dest_path = os.path.join(DATA_RAW_DIR, filename)
        print(f"Fetching {filename} from {url}")
        download_resumable(url, dest_path)

        size = os.path.getsize(dest_path)
        if size < 1024:
            raise RuntimeError(
                f"{filename} downloaded but is suspiciously small ({size} bytes) -- "
                "likely an error page, not the real file. Aborting rather than "
                "silently proceeding with bad data."
            )
        print(f"  saved: {dest_path} ({size:,} bytes)")

    zip_path = os.path.join(DATA_RAW_DIR, "wrinkle_ridges_180.zip")
    extract_dir = os.path.join(DATA_RAW_DIR, "wrinkle_ridges_shapefile")
    if not os.path.exists(extract_dir):
        print(f"Extracting {zip_path} -> {extract_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    else:
        print(f"{extract_dir} already exists, skipping extraction")

    print("Done.")


if __name__ == "__main__":
    main()
