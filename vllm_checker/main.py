from __future__ import annotations

import argparse
from pathlib import Path

from .checker import check_csv_with_llm


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a sam3_realtime_progress CSV with an LLM that "
            "answers per-object yes/no questions and adds a new column "
            "listing objects the LLM still considers missing."
        )
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help=(
            "Path to the input sam3_realtime_progress CSV (e.g. "
            "sam3_realtime_progress.csv)."
        ),
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help=(
            "Optional path for the augmented CSV. Defaults to "
            "<input_stem>_vllm_checked.csv next to the input file."
        ),
    )
    parser.add_argument(
        "--images_root",
        default=None,
        help=(
            "Optional root directory to prepend to each image_path from "
            "the CSV when constructing the full image path."
        ),
    )
    parser.add_argument(
        "--new_column",
        default="objects_vllm_said_no",
        help=(
            "Name of the additional column to write with objects that "
            "the LLM answered 'no' for."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else None
    images_root = Path(args.images_root).resolve() if args.images_root else None

    final_path = check_csv_with_llm(
        input_csv=input_csv,
        output_csv=output_csv,
        images_root=images_root,
        new_column=args.new_column,
    )

    print(f"Wrote LLM-checked CSV to: {final_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

