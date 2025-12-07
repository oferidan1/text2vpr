import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


CSV_REQUIRED_COLUMNS = ["image_path", "description"]


@dataclass
class CaptionRow:
    image_path: str
    description: str


def _normalize_header(name: str) -> str:
    """Normalize CSV headers to a consistent form."""
    return name.strip()


def parse_caption_csv(csv_path: str | Path) -> Iterator[CaptionRow]:
    """
    Parse a clustered CSV and yield CaptionRow objects.

    The input CSV is expected to contain at least the following columns:
    - image_path
    - description
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        # Map normalized header names back to original
        header_map = {_normalize_header(h): h for h in reader.fieldnames}

        missing = [col for col in CSV_REQUIRED_COLUMNS if col not in header_map]
        if missing:
            raise ValueError(
                f"Missing required columns in CSV {csv_path}: {missing}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            image_path = row[header_map["image_path"]]
            description = row[header_map["description"]]
            if image_path is None or description is None:
                # Skip malformed rows instead of failing the whole job
                continue
            yield CaptionRow(
                image_path=str(image_path),
                description=str(description),
            )


