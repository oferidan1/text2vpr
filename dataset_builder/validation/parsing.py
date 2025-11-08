import os
import re
from typing import List

import pandas as pd


PANOID_REGEX = re.compile(r"_([A-Za-z0-9_-]+)\.(?:jpg|JPG)$")


def extract_city_from_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    # Expecting .../Images/<City>/<filename>
    try:
        idx = parts.index("Images")
        return parts[idx + 1]
    except Exception:
        return ""


def extract_panoid_from_path(path: str) -> str:
    m = PANOID_REGEX.search(path)
    return m.group(1) if m else ""


def load_predictions_csv(pred_csv: str) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    if "image_path" not in df.columns or "description" not in df.columns:
        raise ValueError("Predictions CSV must contain 'image_path' and 'description'.")
    df = df.copy()
    df["image_name"] = df["image_path"].apply(lambda p: os.path.basename(str(p)))
    df["city_id"] = df["image_path"].apply(extract_city_from_path)
    df["panoid"] = df["image_path"].apply(extract_panoid_from_path)
    return df


def load_all_assignments(clustered_dir: str, only_city: str = "") -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for name in sorted(os.listdir(clustered_dir)):
        if not name.lower().endswith(".csv"):
            continue
        if name.lower().endswith("_centers.csv"):
            continue
        city = os.path.splitext(name)[0]
        if only_city and city.lower() != only_city.lower():
            continue
        path = os.path.join(clustered_dir, name)
        df = pd.read_csv(path, usecols=["panoid", "cluster_id", "city_id"])
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["panoid", "cluster_id", "city_id"])
    return pd.concat(rows, ignore_index=True)


