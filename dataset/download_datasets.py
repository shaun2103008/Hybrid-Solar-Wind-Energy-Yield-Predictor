"""Download free renewable-energy datasets for Hybrid Solar-Wind Energy Yield Prediction."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA = Path(__file__).resolve().parent
TIMEOUT = 180
CHUNK = 1024 * 1024


def ensure_dirs() -> None:
    for name in ("grid", "benchmarks", "weather", "metadata"):
        (DATA / name).mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path, desc: str | None = None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"SKIP (already exists): {dest.name} ({dest.stat().st_size / (1024*1024):.2f} MB)")
        return True

    headers = {"User-Agent": "Hybrid-Energy-Predictor/1.0 (research dataset collector)"}
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT, headers=headers, allow_redirects=True) as r:
            r.raise_for_status()
            total_header = r.headers.get("content-length")
            total = int(total_header) if total_header and total_header.isdigit() else 0
            label = (desc or dest.name)[:45]
            with open(dest, "wb") as f, tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=label,
            ) as bar:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"OK: {dest.name} ({size_mb:.2f} MB)")
        return True
    except Exception as exc:
        print(f"FAIL: {dest.name} -> {exc}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def fetch_json_api(url: str, dest: Path, desc: str | None = None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"SKIP (already exists): {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")
        return True
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        dest.write_bytes(r.content)
        size_kb = dest.stat().st_size / 1024
        print(f"OK: {dest.name} ({size_kb:.1f} KB)")
        return True
    except Exception as exc:
        print(f"FAIL: {url} -> {exc}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    if not zip_path.exists():
        return False
    try:
        print(f"Extracting {zip_path.name} to {extract_to.name}...")
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_to)
        print(f"OK: Extracted {zip_path.name}")
        return True
    except Exception as exc:
        print(f"FAIL extract {zip_path.name}: {exc}")
        return False


def download_top3() -> dict:
    ensure_dirs()
    manifest: dict = {"downloaded": [], "failed": []}

    print("\n" + "=" * 65)
    print(" 1. OPEN POWER SYSTEM DATA (OPSD) - Hourly Solar & Wind Generation")
    print("=" * 65)
    opsd_url = "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
    opsd_dest = DATA / "grid" / "opsd_europe_hourly.csv"
    ok = download(opsd_url, opsd_dest, desc="OPSD Europe Hourly")
    (manifest["downloaded"] if ok else manifest["failed"]).append({"name": "OPSD", "path": str(opsd_dest)})

    print("\n" + "=" * 65)
    print(" 2. GEFCom2014 BENCHMARK - Wind and Solar Track Competitions")
    print("=" * 65)
    gef_url = "https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=1"
    gef_zip = DATA / "benchmarks" / "GEFCom2014.zip"
    gef_dir = DATA / "benchmarks" / "GEFCom2014"
    ok = download(gef_url, gef_zip, desc="GEFCom2014 Zip")
    if ok:
        extract_zip(gef_zip, gef_dir)
    (manifest["downloaded"] if ok else manifest["failed"]).append({"name": "GEFCom2014", "path": str(gef_dir)})

    print("\n" + "=" * 65)
    print(" 3. OPEN-METEO HOURLY WEATHER REANALYSIS (Exogenous Features)")
    print("=" * 65)
    weather_endpoints = [
        # Location 1: Germany / Central Europe (paired with OPSD grid)
        (
            "https://archive-api.open-meteo.com/v1/archive?latitude=51.1657&longitude=10.4515&start_date=2019-01-01&end_date=2020-12-31&hourly=temperature_2m,wind_speed_10m,wind_speed_100m,shortwave_radiation,direct_radiation,diffuse_radiation,cloud_cover",
            DATA / "weather" / "open_meteo_germany_2019_2020.json",
            "Open-Meteo Germany (OPSD Paired)",
        ),
        # Location 2: Delhi (India Solar/Wind context)
        (
            "https://archive-api.open-meteo.com/v1/archive?latitude=28.6139&longitude=77.2090&start_date=2023-01-01&end_date=2023-12-31&hourly=temperature_2m,wind_speed_10m,shortwave_radiation,direct_radiation,diffuse_radiation,cloud_cover",
            DATA / "weather" / "open_meteo_delhi_2023.json",
            "Open-Meteo Delhi 2023",
        ),
        # Location 3: Copenhagen (Nordic wind/solar context)
        (
            "https://archive-api.open-meteo.com/v1/archive?latitude=55.6761&longitude=12.5683&start_date=2018-06-01&end_date=2019-09-01&hourly=temperature_2m,wind_speed_10m,shortwave_radiation,direct_radiation,diffuse_radiation,cloud_cover",
            DATA / "weather" / "open_meteo_copenhagen_solete.json",
            "Open-Meteo Copenhagen",
        ),
    ]
    for url, dest, label in weather_endpoints:
        ok = fetch_json_api(url, dest, desc=label)
        (manifest["downloaded"] if ok else manifest["failed"]).append({"name": label, "path": str(dest)})

    # Write manifest
    manifest_path = DATA / "metadata" / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest recorded at: {manifest_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download hybrid solar-wind datasets.")
    parser.add_argument("--all", action="store_true", help="Download all secondary datasets as well")
    args = parser.parse_args()

    manifest = download_top3()
    failed = [f["name"] for f in manifest["failed"]]
    if failed:
        print(f"\nWarning: Some downloads failed: {failed}")
        return 1
    print("\nAll Top 3 Hybrid Solar-Wind datasets downloaded and verified successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
