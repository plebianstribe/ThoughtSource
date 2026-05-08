#!/usr/bin/env python3
"""
Extract question, answer, generated_cot, and cot_correctness from ThoughtSource datasets.

For each dataset the script prints the file path where the data is loaded from and
writes a flat CSV with one row per (item × generated_cot entry).

Fields exported
---------------
source_file       : path to the JSON file the item was read from
dataset           : dataset name (e.g. commonsense_qa, worldtree, …)
split             : train / validation / test
item_id           : unique item id
question          : the question text
answer            : gold answer(s) as a JSON list
generated_cot     : the chain-of-thought text produced by a model
cot_model         : model name that produced the CoT
cot_trigger       : prompting strategy used (e.g. kojima-01, zhou-01)
cot_author        : author / source of the CoT generation run
correct_answer_auto   : automatic correctness flag (bool) – does the extracted answer match the gold?
cot_correctness_human : human-evaluated quality annotations as a JSON dict
                        (keys: 'Incorrect reading comprehension', 'Incorrect reasoning',
                         'Insufficient knowledge', 'Too verbose', 'comment', 'preferred', …)
                        None when no human annotation is present.

Data locations
--------------
Pre-built collections (primary source):
  libs/cot/cot/datasets/thoughtsource/thoughtsource_33.json
  libs/cot/cot/datasets/thoughtsource/thoughtsource_100.json
  libs/cot/cot/datasets/thoughtsource/thoughtsource_33_paper.json

Human-annotated files (contain cot_correctness_human values):
  notebooks/internal_documentation/annotated_files/*.json
  notebooks/internal_documentation/newly_annotated/*.json

Usage
-----
  # Extract from the default pre-built collections (thoughtsource_33 + thoughtsource_100):
  python extract_cot_data.py

  # Also include the annotated files:
  python extract_cot_data.py --include-annotated

  # Extract from a specific JSON file:
  python extract_cot_data.py --file path/to/your_collection.json

  # Change the output CSV path (default: cot_data_extracted.csv):
  python extract_cot_data.py --output my_output.csv
"""

import argparse
import csv
import json
import os
import pathlib
import sys

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).parent.absolute()

PREBUILT_COLLECTIONS = [
    REPO_ROOT / "libs" / "cot" / "cot" / "datasets" / "thoughtsource" / "thoughtsource_33.json",
    REPO_ROOT / "libs" / "cot" / "cot" / "datasets" / "thoughtsource" / "thoughtsource_100.json",
    REPO_ROOT / "libs" / "cot" / "cot" / "datasets" / "thoughtsource" / "thoughtsource_33_paper.json",
]

ANNOTATED_DIRS = [
    REPO_ROOT / "notebooks" / "internal_documentation" / "annotated_files",
    REPO_ROOT / "notebooks" / "internal_documentation" / "newly_annotated",
]

CSV_COLUMNS = [
    "source_file",
    "dataset",
    "split",
    "item_id",
    "question",
    "answer",
    "generated_cot",
    "cot_model",
    "cot_trigger",
    "cot_author",
    "correct_answer_auto",
    "cot_correctness_human",
]


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------

def _parse_model_name(model_str: str) -> str:
    """Return the model name from the serialised model dict string."""
    if not model_str:
        return ""
    try:
        model_dict = json.loads(model_str)
        return model_dict.get("name", model_str)
    except (json.JSONDecodeError, TypeError):
        # some entries use Python repr instead of JSON
        try:
            import ast
            model_dict = ast.literal_eval(model_str)
            return model_dict.get("name", model_str)
        except Exception:
            return model_str


def extract_records_from_data(data: dict, source_file: str) -> list:
    """
    Given a loaded ThoughtSource JSON dict (keyed by dataset → split → [items]),
    return a flat list of record dicts ready for CSV output.
    """
    records = []
    for dataset_name, dataset_splits in data.items():
        if not isinstance(dataset_splits, dict):
            continue
        for split_name, rows in dataset_splits.items():
            if not isinstance(rows, list):
                continue
            for item in rows:
                question = item.get("question", "")
                answer = json.dumps(item.get("answer", []))

                generated_cots = item.get("generated_cot", [])
                if not generated_cots:
                    # Emit one row even when no CoT is present
                    records.append({
                        "source_file": source_file,
                        "dataset": dataset_name,
                        "split": split_name,
                        "item_id": item.get("id", ""),
                        "question": question,
                        "answer": answer,
                        "generated_cot": "",
                        "cot_model": "",
                        "cot_trigger": "",
                        "cot_author": "",
                        "correct_answer_auto": None,
                        "cot_correctness_human": None,
                    })
                    continue

                for gcot_entry in generated_cots:
                    cot_text = gcot_entry.get("cot", "")
                    cot_trigger = gcot_entry.get("cot_trigger", "")
                    cot_author = gcot_entry.get("author", "")
                    cot_model = _parse_model_name(gcot_entry.get("model", ""))

                    # Automatic correctness: from the first extracted answer
                    answers_list = gcot_entry.get("answers", [])
                    correct_answer_auto = None
                    if answers_list:
                        correct_answer_auto = answers_list[0].get("correct_answer")

                    # Human-evaluated correctness: from annotations
                    annotations = gcot_entry.get("annotations", [])
                    cot_correctness_human = None
                    if annotations:
                        human_eval = {
                            ann["key"]: ann["value"]
                            for ann in annotations
                            if "key" in ann and "value" in ann
                        }
                        if human_eval:
                            cot_correctness_human = json.dumps(human_eval)

                    records.append({
                        "source_file": source_file,
                        "dataset": dataset_name,
                        "split": split_name,
                        "item_id": item.get("id", ""),
                        "question": question,
                        "answer": answer,
                        "generated_cot": cot_text,
                        "cot_model": cot_model,
                        "cot_trigger": cot_trigger,
                        "cot_author": cot_author,
                        "correct_answer_auto": correct_answer_auto,
                        "cot_correctness_human": cot_correctness_human,
                    })
    return records


def load_json_file(path: pathlib.Path) -> dict:
    """Load and return a ThoughtSource JSON collection file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_annotated_json_files(dirs: list) -> list:
    """Return all valid JSON file paths found inside the given directories."""
    paths = []
    for d in dirs:
        if not d.exists():
            continue
        for entry in sorted(d.iterdir()):
            if entry.is_file() and entry.suffix == ".json":
                paths.append(entry)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract question/answer/generated_cot/cot_correctness from ThoughtSource datasets."
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Path to a specific ThoughtSource JSON collection file to extract from.",
    )
    parser.add_argument(
        "--include-annotated",
        action="store_true",
        default=False,
        help=(
            "Also extract from the human-annotated files in "
            "notebooks/internal_documentation/annotated_files/ and newly_annotated/."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="CSV_PATH",
        default="cot_data_extracted.csv",
        help="Output CSV file path (default: cot_data_extracted.csv).",
    )
    args = parser.parse_args()

    # Build list of source files to process
    source_files: list[pathlib.Path] = []

    if args.file:
        p = pathlib.Path(args.file).resolve()
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)
        source_files.append(p)
    else:
        for p in PREBUILT_COLLECTIONS:
            if p.exists():
                source_files.append(p)
            else:
                print(f"WARNING: pre-built collection not found, skipping: {p}", file=sys.stderr)

    if args.include_annotated:
        annotated_files = collect_annotated_json_files(ANNOTATED_DIRS)
        source_files.extend(annotated_files)

    if not source_files:
        print("ERROR: No source files found to process.", file=sys.stderr)
        sys.exit(1)

    # Print data sources
    print("\n=== Data source files ===")
    for p in source_files:
        print(f"  {p}")
    print()

    # Extract records from all source files
    all_records = []
    for p in source_files:
        print(f"Loading {p.name} ...", end=" ", flush=True)
        try:
            data = load_json_file(p)
        except json.JSONDecodeError as exc:
            print(f"SKIPPED (JSON parse error: {exc})")
            continue
        except OSError as exc:
            print(f"SKIPPED (OS error: {exc})")
            continue

        records = extract_records_from_data(data, str(p))
        print(f"{len(records)} records extracted.")
        all_records.extend(records)

    print(f"\nTotal records extracted: {len(all_records)}")

    # Write CSV
    output_path = pathlib.Path(args.output).resolve()
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"CSV written to: {output_path}")

    # Summary by dataset
    print("\n=== Summary by dataset ===")
    summary: dict[tuple[str, str], int] = {}
    for rec in all_records:
        key = (rec["source_file"].split(os.sep)[-1], rec["dataset"])
        summary[key] = summary.get(key, 0) + 1
    for (src_file, dataset_name), count in sorted(summary.items()):
        has_human = sum(
            1 for r in all_records
            if r["source_file"].endswith(src_file)
            and r["dataset"] == dataset_name
            and r["cot_correctness_human"] is not None
        )
        human_note = f" ({has_human} with human annotations)" if has_human else ""
        print(f"  [{src_file}] {dataset_name}: {count} records{human_note}")


if __name__ == "__main__":
    main()
