#!/usr/bin/env python3
"""
Generate a teaching dataset for the lecture
"Datenvorbereitung mit Python: Vom Rohdatensatz zum trainierbaren Modell".

What the script does:
1. Downloads labeled images from Wikimedia Commons for
   - Black-headed Gull (Lachmöwe) / Chroicocephalus ridibundus
   - European Herring Gull (Silbermöwe) / Larus argentatus
2. Stores image provenance and licensing metadata.
3. Creates a feature table that represents features extracted from the images.
   Some features are derived from the downloaded images. Other morphology-like
   features are simulated to mimic a preceding computer-vision feature extraction
   stage while keeping the teaching dataset controllable.
4. Injects controlled data-quality problems:
   - missing values
   - missing labels
   - mixed data types
   - categorical variables
   - outliers
   - different feature scales
   - redundant features
   - irrelevant features
5. Writes both a clean "source" CSV and a deliberately messy "raw" CSV.

The resulting dataset is for teaching/demo purposes. The simulated morphology
features are NOT scientific measurements of the downloaded birds.

Requirements:
    pip install requests pandas numpy pillow scikit-learn

Usage:
    python generate_moewe_dataset.py --images-per-class 40 --seed 42
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageStat

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MoeweLectureDataset/1.0 (educational use; contact@example.invalid)"

SPECIES = {
    "lachmoewe": {
        "label": "Lachmöwe",
        "common": "Black-headed Gull",
        "scientific": "Chroicocephalus ridibundus",
        "category": "Category:Chroicocephalus ridibundus",
    },
    "silbermoewe": {
        "label": "Silbermöwe",
        "common": "European Herring Gull",
        "scientific": "Larus argentatus",
        "category": "Category:Larus argentatus",
    },
}


class CommonsError(RuntimeError):
    pass


def commons_api(params: Dict[str, str], timeout: int = 30) -> dict:
    params = dict(params)
    params["format"] = "json"
    params["formatversion"] = "2"
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(API_URL, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise CommonsError(str(data["error"]))
    return data


def list_category_files(category: str, limit: int) -> List[dict]:
    """Return file metadata for images directly in a Commons category."""
    results: List[dict] = []
    cont: Dict[str, str] = {}

    while len(results) < limit:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo|info",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": "1600",
        }
        params.update(cont)
        data = commons_api(params)
        pages = data.get("query", {}).get("pages", [])
        for page in pages:
            infos = page.get("imageinfo", [])
            if not infos:
                continue
            info = infos[0]
            mime = info.get("mime", "")
            if not mime.startswith("image/"):
                continue
            title = page.get("title", "")
            # Avoid obvious non-photo items.
            if title.lower().endswith((".svg", ".gif", ".tif", ".tiff")):
                continue
            results.append(page)
            if len(results) >= limit:
                break

        if len(results) >= limit:
            break
        cont = data.get("continue", {})
        if not cont:
            break
        time.sleep(0.15)

    return results


def clean_extmetadata(extmeta: dict, key: str) -> str:
    value = extmeta.get(key, {}) if extmeta else {}
    raw = value.get("value", "") if isinstance(value, dict) else str(value)
    # Strip basic HTML tags from Commons metadata.
    return re.sub(r"<[^>]+>", "", raw).strip()


def choose_image_url(page: dict) -> str:
    info = page["imageinfo"][0]
    return info.get("thumburl") or info.get("url")


def image_metadata(page: dict) -> dict:
    info = page["imageinfo"][0]
    ext = info.get("extmetadata", {})
    return {
        "commons_title": page.get("title", ""),
        "source_url": info.get("descriptionurl", ""),
        "file_url": info.get("url", ""),
        "author": clean_extmetadata(ext, "Artist"),
        "license": clean_extmetadata(ext, "LicenseShortName"),
        "license_url": clean_extmetadata(ext, "LicenseUrl"),
        "date_taken": clean_extmetadata(ext, "DateTimeOriginal"),
        "image_width": info.get("width"),
        "image_height": info.get("height"),
        "mime": info.get("mime", ""),
    }


def download_image(session: requests.Session, url: str, path: Path) -> None:
    response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def image_features(path: Path) -> Dict[str, float]:
    """Extract simple image-derived features. These are intentionally generic."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        width, height = im.size
        # Resize for stable low-cost statistics.
        thumb = im.copy()
        thumb.thumbnail((512, 512))
        arr = np.asarray(thumb, dtype=np.float32) / 255.0

    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    helligkeit = float(gray.mean())
    helligkeit_std = float(gray.std())
    mittlerer_rotanteil = float(arr[..., 0].mean())
    mittlerer_gruenanteil = float(arr[..., 1].mean())
    mittlerer_blauanteil = float(arr[..., 2].mean())

    # Very rough saturation proxy.
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    saettigung = float(
        np.where(
            mx == 0,
            0,
            (mx - mn) / np.maximum(mx, 1e-6),
        ).mean()
    )

    # Approximate horizontal/vertical gradient magnitude using finite differences.
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    kantendichte = float(
        (np.abs(gx).mean() + np.abs(gy).mean()) / 2.0
    )

    return {
        "bildbreite_px": float(width),
        "bildhoehe_px": float(height),
        "bildhelligkeit": helligkeit,
    }


def simulated_extracted_features(
    label: str,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Simulate morphology/vision features extracted from the image.

    The class distributions overlap deliberately. This is not a biological model.
    """
    is_herring = label == "Silbermöwe"

    # Means chosen only for didactic plausibility and overlap.
    if is_herring:
        fluegellaenge = rng.normal(123.0, 7.0)
        schnabellaenge = rng.normal(46.0, 3.2)
        koerpermasse = rng.normal(620.0, 90.0)
        kopfflaeche = rng.normal(12800, 1800)
        fluegelflaeche = rng.normal(26500, 4200)
        kopfhelligkeit = rng.normal(0.70, 0.08)
    else:
        fluegellaenge = rng.normal(109.0, 6.5)
        schnabellaenge = rng.normal(40.0, 2.8)
        kopfflaeche = rng.normal(10800, 1600)
        fluegelflaeche = rng.normal(21800, 3600)
        kopfhelligkeit = rng.normal(0.77, 0.07)

    return {
        "fluegellaenge_mm": max(80.0, fluegellaenge),
        "schnabellaenge_mm": max(20.0, schnabellaenge),
        "kopfflaeche_px": max(2000.0, kopfflaeche),
        "fluegelflaeche_px": max(5000.0, fluegelflaeche),
        "kopfhelligkeit": float(
            np.clip(kopfhelligkeit, 0.2, 1.0)
        ),
        "schnabelfarbwert": float(
            np.clip(
                rng.normal(
                    0.68 if is_herring else 0.58,
                    0.12,
                ),
                0.0,
                1.0,
            )
        ),
    }


def build_clean_dataset(records: List[dict], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[dict] = []

    for record in records:
        label = record["label"]

        row = {
            "bild_id": record["bild_id"],
            "art": label,
            "geschlecht": rng.choice(
                ["weiblich", "maennlich"],
                p=[0.5, 0.5],
            ),
            "schnabelfarbe": rng.choice(
                ["gelb", "orange", "rot"],
                p=[0.55, 0.35, 0.10],
            ),
            "kamera_id": rng.choice(
                ["kamera_A", "kamera_B", "kamera_C"]
            ),
            "bildqualitaet": rng.choice(
                ["gut", "mittel", "schlecht"],
                p=[0.65, 0.28, 0.07],
            ),
            "beobachtungsort": rng.choice(
                ["Nordsee", "Ostsee", "Binnenland"],
                p=[0.55, 0.25, 0.20],
            ),
        }

        row.update(record["img_features"])
        row.update(
            simulated_extracted_features(
                label,
                rng,
            )
        )

        rows.append(row)

    df = pd.DataFrame(rows)


    # Explicitly introduce some visually plausible but non-biological metadata.
    # These are useful for teaching data quality and leakage questions.
    return df


def inject_messiness(
    df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 10_000)
    raw = df.copy()

    n = len(raw)

    # 1) Missing feature values: concentrated in a few columns.
    for col, frac in {
        "schnabellaenge_mm": 0.06,
        "schnabelfarbe": 0.04,
        "geschlecht": 0.08,
    }.items():
        count = max(1, int(round(n * frac)))
        idx = rng.choice(
            raw.index.to_numpy(),
            size=count,
            replace=False,
        )
        raw.loc[idx, col] = np.nan

    # 2) Missing labels.
    count = max(2, int(round(n * 0.025)))
    idx = rng.choice(
        raw.index.to_numpy(),
        size=count,
        replace=False,
    )
    raw.loc[idx, "art"] = np.nan

    # 3) Mixed data types in selected numeric columns.
    numeric_col = "schnabellaenge_mm"
    raw[numeric_col] = raw[numeric_col].astype(object)

    idx = rng.choice(
        raw.index.to_numpy(),
        size=max(2, int(n * 0.03)),
        replace=False,
    )

    values = raw.loc[idx, numeric_col]

    raw.loc[idx, numeric_col] = values.map(
        lambda x: (
            str(round(float(x), 1))
            if pd.notna(x)
            else x
        )
    )

    # 4) Categorical inconsistency.
    #idx = rng.choice(
    #    raw.index.to_numpy(),
    #    size=max(1, int(n * 0.02)),
    #    replace=False,
    #)

    #raw.loc[idx, "schnabelfarbe"] = "unbekannt"

    # 5) Outliers. Keep only a handful.
    for col, values in {
        "fluegellaenge_mm": [178.0],
        "kopfflaeche_px": [65000.0],
    }.items():
        idx = int(
            rng.choice(
                raw.index.to_numpy()
            )
        )

        raw.loc[idx, col] = float(
            rng.choice(values)
        )

    # 6) Measurement inconsistencies in one duplicated feature.
    # Keep the redundant body_mass_kg mostly consistent but create one obvious unit error.
    idx = int(
        rng.choice(
            raw.index.to_numpy()
        )
    )

    # All existing camera A assignments belonging to Silbermöwen
    # are reassigned to another camera.
    silver_mask = raw["art"].eq("Silbermöwe")

    silver_camera_a = (
        silver_mask
        & raw["kamera_id"].eq("kamera_A")
    )

    n_reassign = silver_camera_a.sum()

    if n_reassign > 0:
        replacement_cameras = rng.choice(
            ["kamera_B", "kamera_C"],
            size=n_reassign,
            replace=True,
        )

        raw.loc[
            silver_camera_a,
            "kamera_id"
        ] = replacement_cameras

    # Assign a substantial fraction of Lachmöwen to camera A.
    # Only rows with a known label are considered.
    gull_mask = raw["art"].eq("Lachmöwe")

    gull_idx = raw.index[gull_mask].to_numpy()

    if len(gull_idx) > 0:
        n_camera_a = max(
            2,
            int(round(len(gull_idx) * 0.5))
        )

        chosen = rng.choice(
            gull_idx,
            size=min(n_camera_a, len(gull_idx)),
            replace=False,
        )

        raw.loc[
            chosen,
            "kamera_id"
        ] = "kamera_A"
    # 8) Shuffle rows.
    raw = (
        raw
        .sample(
            frac=1.0,
            random_state=seed,
        )
        .reset_index(drop=True)
    )

    return raw


def make_downloads(
    outdir: Path,
    images_per_class: int,
    seed: int,
) -> Tuple[List[dict], pd.DataFrame]:
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images_dir = outdir / "images"
    images_dir.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {"User-Agent": USER_AGENT}
    )

    rng = random.Random(seed)

    all_records: List[dict] = []
    provenance_rows: List[dict] = []

    for species_key, spec in SPECIES.items():

        print(
            f"Fetching Commons files for "
            f"{spec['common']} ..."
        )

        pages = list_category_files(
            spec["category"],
            max(
                images_per_class * 3,
                images_per_class + 10,
            ),
        )

        rng.shuffle(pages)

        selected = []
        seen_names = set()

        for page in pages:
            title = page.get("title", "")

            if title in seen_names:
                continue

            seen_names.add(title)
            selected.append(page)

            if len(selected) >= images_per_class:
                break

        if len(selected) < images_per_class:
            raise CommonsError(
                f"Only found {len(selected)} usable images "
                f"for {spec['common']}; "
                f"requested {images_per_class}."
            )

        for i, page in enumerate(
            selected,
            start=1,
        ):
            meta = image_metadata(page)
            source_url = choose_image_url(page)

            safe_name = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                page.get("title", "file"),
            )

            ext = (
                Path(safe_name)
                .suffix
                .lower()
                or ".jpg"
            )

            image_id = (
                f"{species_key}_{i:03d}"
            )

            filename = (
                f"{image_id}{ext}"
            )

            filepath = images_dir / filename

            try:
                if not filepath.exists():
                    download_image(
                        session,
                        source_url,
                        filepath,
                    )
                    time.sleep(0.15)

                bildmerkmale = image_features(
                    filepath
                )

            except Exception as exc:
                print(
                    f"  Skipping "
                    f"{page.get('title')}: "
                    f"{exc}"
                )
                continue

            record = {
                "bild_id": image_id,
                "label": spec["label"],
                "species_key": species_key,
                "filepath": str(
                    filepath.as_posix()
                ),
                "img_features": bildmerkmale,
            }

            all_records.append(record)

            provenance_rows.append(
                {
                    "bild_id": image_id,
                    "art": spec["label"],
                    "scientific_name": spec["scientific"],
                    "local_file": str(
                        filepath.as_posix()
                    ),
                    **meta,
                }
            )

    provenance = (
        pd.DataFrame(provenance_rows)
        .sort_values("bild_id")
        .reset_index(drop=True)
    )

    return all_records, provenance


def save_json_config(
    outdir: Path,
    seed: int,
    images_per_class: int,
) -> None:
    config = {
        "seed": seed,
        "images_per_class": images_per_class,
        "source": "Wikimedia Commons",
        "species": {
            "Lachmöwe":
                "Chroicocephalus ridibundus",
            "Silbermöwe":
                "Larus argentatus",
        },
        "note": (
            "Morphology-like features are synthetic "
            "teaching features and image features "
            "are extracted from downloaded images."
        ),
    }

    (
        outdir / "dataset_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--images-per-class",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("moewen_ml_demo"),
    )

    args = parser.parse_args()

    if args.images_per_class < 5:
        raise SystemExit(
            "Please request at least 5 images per class."
        )

    np.random.seed(args.seed)
    random.seed(args.seed)

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Output directory: "
        f"{args.outdir.resolve()}"
    )

    records, provenance = make_downloads(
        args.outdir,
        args.images_per_class,
        args.seed,
    )

    if not records:
        raise SystemExit(
            "No images could be downloaded."
        )

    clean = build_clean_dataset(
        records,
        args.seed,
    )

    raw = inject_messiness(
        clean,
        args.seed,
    )

    clean.to_csv(
        args.outdir
        / "moewen_clean_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw.to_csv(
        args.outdir
        / "moewen_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    provenance.to_csv(
        args.outdir
        / "image_provenance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json_config(
        args.outdir,
        args.seed,
        args.images_per_class,
    )

    print()

    print(
        f"Downloaded usable images: "
        f"{len(records)} "
        f"({len(records)//2} target minimum each class approximately)"
    )

    print(
        f"Clean source CSV: "
        f"{args.outdir / 'moewen_clean_source.csv'}"
    )

    print(
        f"Messy teaching CSV: "
        f"{args.outdir / 'moewen_raw.csv'}"
    )

    print(
        f"Image provenance: "
        f"{args.outdir / 'image_provenance.csv'}"
    )

    print(
        f"Images: "
        f"{args.outdir / 'images'}"
    )

    print()

    print(
        "Important: The morphology-like features "
        "are synthetic teaching features and should "
        "not be presented as real measurements."
    )


if __name__ == "__main__":
    main()