from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from tqdm import tqdm

from .object_utils import parse_objects_field, join_objects
from .llm_client import TextOnlyLLMClient, build_default_client


@dataclass
class ProgressRow:
    """Represents one row from `sam3_realtime_progress*.csv`.

    Only the fields we actually use are modeled explicitly; all
    remaining columns are preserved in a generic dict when rewriting
    the CSV so the script is robust to schema changes.
    """

    raw: dict  # Full original row as a dict
    image_path: str
    description: str
    objects_not_found: str


def iter_progress_rows(csv_path: Path) -> Iterable[ProgressRow]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            description = (row.get("description") or "").strip()
            objects_not_found = (row.get("objects_not_found") or "").strip()

            if not image_path:
                # Be forgiving and just skip malformed rows.
                continue

            yield ProgressRow(
                raw=row,
                image_path=image_path,
                description=description,
                objects_not_found=objects_not_found,
            )


def check_csv_with_llm(
    *,
    input_csv: Path,
    output_csv: Optional[Path] = None,
    images_root: Optional[Path] = None,
    client: Optional[TextOnlyLLMClient] = None,
    new_column: str = "objects_vllm_said_no",
) -> Path:
    """Run LLM-based checks over a SAM3 realtime progress CSV.

    For each row where `objects_not_found` is non-empty, we split that
    field into individual objects (dot-separated), ask the LLM a
    yes/no question for each object and image, and write a new CSV
    that copies all original columns plus an extra column containing
    the subset of objects for which the LLM answered "no".
    """

    input_csv = input_csv.resolve()
    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_vllm_checked.csv")
    else:
        output_csv = output_csv.resolve()

    if images_root is not None:
        images_root = images_root.resolve()

    client = client or build_default_client()

    # Open input once to access header order.
    with input_csv.open("r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {input_csv}")

        fieldnames: List[str] = list(reader.fieldnames)
        if new_column not in fieldnames:
            fieldnames.append(new_column)

        with output_csv.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            f_out.flush()

            for row in tqdm(reader, desc="LLM-checking rows", unit="row"):
                pr = ProgressRow(
                    raw=row,
                    image_path=(row.get("image_path") or "").strip(),
                    description=(row.get("description") or "").strip(),
                    objects_not_found=(row.get("objects_not_found") or "").strip(),
                )

                if not pr.image_path:
                    # Keep malformed rows but leave new column empty.
                    row[new_column] = ""
                    writer.writerow(row)
                    f_out.flush()
                    continue

                # Build full image path using the provided root directory.
                full_image_path: str
                if images_root is not None:
                    full_image_path = str((images_root / pr.image_path).resolve())
                else:
                    full_image_path = pr.image_path

                missing_objects = parse_objects_field(pr.objects_not_found)
                if not missing_objects:
                    row[new_column] = ""
                    writer.writerow(row)
                    f_out.flush()
                    continue

                llm_rejected: List[str] = []
                for obj in missing_objects:
                    present = client.is_object_in_image(
                        image_path=full_image_path,
                        object_name=obj,
                        description=pr.description,
                    )
                    # We keep objects for which the model said "no".
                    if not present:
                        llm_rejected.append(obj)

                row[new_column] = join_objects(llm_rejected)
                writer.writerow(row)
                f_out.flush()

    return output_csv

