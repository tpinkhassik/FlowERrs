#!/usr/bin/env python3
import argparse
import hashlib
import os
import sys
from typing import TextIO


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream-split a large reaction file into train/val/test using a deterministic "
            "random mapping based on the final pipe-delimited reaction id field."
        )
    )
    parser.add_argument("--input", required=True, help="Input reactions file")
    parser.add_argument("--train-output", required=True, help="Output path for training split")
    parser.add_argument("--val-output", required=True, help="Output path for validation split")
    parser.add_argument("--test-output", required=True, help="Output path for test split")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic split seed (default: 0)",
    )
    parser.add_argument(
        "--strict-id",
        action="store_true",
        help="Require the final field to parse as an integer id; otherwise skip line",
    )
    return parser.parse_args()


def _bucket_from_reaction_id(reaction_id: str, seed: int) -> int:
    key = f"{seed}:{reaction_id}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % 10000


def _extract_reaction_id(line: str) -> str:
    stripped = line.rstrip("\n")
    _, reaction_id = stripped.rsplit("|", 1)
    return reaction_id.strip()


def _open_writer(path: str) -> TextIO:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return open(path, "w", encoding="utf-8", newline="")


def main():
    args = parse_args()

    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0, "total": 0}

    with open(args.input, "r", encoding="utf-8", errors="replace") as fin, \
            _open_writer(args.train_output) as ftrain, \
            _open_writer(args.val_output) as fval, \
            _open_writer(args.test_output) as ftest:

        for line_num, line in enumerate(fin, start=1):
            counts["total"] += 1
            if not line.strip():
                counts["skipped"] += 1
                continue

            try:
                reaction_id = _extract_reaction_id(line)
                if args.strict_id:
                    int(reaction_id)
            except Exception:
                counts["skipped"] += 1
                if counts["skipped"] <= 10:
                    sys.stderr.write(
                        f"Skipping malformed line {line_num}: unable to parse reaction id\n"
                    )
                continue

            bucket = _bucket_from_reaction_id(reaction_id, args.seed)
            if bucket < 8900:
                ftrain.write(line)
                counts["train"] += 1
            elif bucket < 9000:
                fval.write(line)
                counts["val"] += 1
            else:
                ftest.write(line)
                counts["test"] += 1

    assigned = counts["train"] + counts["val"] + counts["test"]
    if assigned > 0:
        train_pct = 100.0 * counts["train"] / assigned
        val_pct = 100.0 * counts["val"] / assigned
        test_pct = 100.0 * counts["test"] / assigned
    else:
        train_pct = val_pct = test_pct = 0.0

    print("Split complete.")
    print(f"Input lines: {counts['total']}")
    print(f"Assigned: {assigned}")
    print(f"Skipped: {counts['skipped']}")
    print(f"Train: {counts['train']} ({train_pct:.2f}%)")
    print(f"Val:   {counts['val']} ({val_pct:.2f}%)")
    print(f"Test:  {counts['test']} ({test_pct:.2f}%)")


if __name__ == "__main__":
    main()
