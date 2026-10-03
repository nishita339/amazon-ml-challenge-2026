#!/usr/bin/env python3
"""Inspect labeled training links omitted by the candidate retriever."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import run


def load_missing_records(paths: list[Path], wanted: set[str]) -> dict[str, dict[str, str]]:
    found = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["entity_id"] in wanted:
                    found[row["entity_id"]] = row
                    if len(found) == len(wanted):
                        return found
    return found


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=2000)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=919)
    parser.add_argument("--examples", type=int, default=25)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    train_dir = root / "dataset" / "train"
    target_paths = [train_dir / "train_source2.tsv", train_dir / "train_source3.tsv"]
    work_dir = args.work_dir or root / "work"
    db = run.open_index(work_dir / "train_index.sqlite", target_paths, False)
    try:
        sample = run.sample_source1(train_dir / "train_source1.tsv", args.entities, args.seed)
        truth = run.load_ground_truth(
            train_dir / "train_ground_truth.tsv", {row["entity_id"] for row in sample}
        )
        missed = []
        possible = covered = 0
        for row in sample:
            actual = truth.get(row["entity_id"], set())
            candidates = run.retrieve(db, row, args.candidate_limit)
            candidate_ids = {candidate[0] for candidate in candidates}
            possible += len(actual)
            covered += len(actual & candidate_ids)
            context = run.prepare_entity(row)
            for target_id in actual - candidate_ids:
                missed.append((row, target_id, context))
    finally:
        db.close()

    records = load_missing_records(target_paths, {target_id for _, target_id, _ in missed})
    feature_rows = []
    for row, target_id, context in missed:
        target = records.get(target_id)
        if target:
            candidate = (
                target_id,
                run.normalize(target["business_name"]),
                run.normalize(target["business_address"], True),
            )
            feature_rows.append((row, target, run.pair_features(context, candidate)))

    print(f"sampled_entities={len(sample)}")
    print(f"labeled_target_ids={possible}")
    print(f"retrieved_target_ids={covered}")
    print(f"missed_target_ids={len(missed)}")
    print(f"candidate_recall={covered / possible if possible else 0:.6f}")
    print(f"missed_pairs_found={len(feature_rows)}")
    if feature_rows:
        checks = {
            "name_token_overlap": lambda f: f[0] > 0,
            "name_trigram_overlap": lambda f: f[2] > 0,
            "address_token_overlap": lambda f: f[3] > 0,
            "shared_address_number": lambda f: f[15] > 0,
            "exact_normalized_name": lambda f: f[6] > 0,
            "exact_normalized_address": lambda f: f[7] > 0,
        }
        for label, predicate in checks.items():
            count = sum(predicate(features) for _, _, features in feature_rows)
            print(f"{label}={count}/{len(feature_rows)} ({count / len(feature_rows):.1%})")

        print("\nRepresentative missed matches:")
        for row, target, features in feature_rows[:args.examples]:
            print(f"S1: {row['entity_id']} | {row['business_name']} | {row['business_address']}")
            print(f"GT: {target['entity_id']} | {target['business_name']} | {target['business_address']}")
            print(
                "sim: "
                f"name_tok={features[0]:.2f} name_tri={features[2]:.2f} "
                f"addr_tok={features[3]:.2f} digits={features[15]:.0f} "
                f"name_ratio={features[11]:.2f} name_sort={features[12]:.2f} "
                f"addr_sort={features[13]:.2f}"
            )


if __name__ == "__main__":
    main()
