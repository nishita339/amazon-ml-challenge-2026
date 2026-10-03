#!/usr/bin/env python3
"""Cross-validate simple probability-threshold policies on the held-out sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from run import load_ground_truth, open_index, pair_features, prepare_entity, retrieve, sample_source1


def score_rows(rows, threshold_s2: float, threshold_s3: float) -> float:
    total = 0.0
    for actual, ids, is_s2, probabilities in rows:
        selected = (probabilities >= np.where(is_s2, threshold_s2, threshold_s3))
        predicted_count = int(selected.sum())
        if not actual:
            total += float(predicted_count == 0)
            continue
        if predicted_count == 0:
            continue
        true_positive = sum(
            1 for entity_id, keep in zip(ids, selected) if keep and entity_id in actual
        )
        if true_positive:
            total += 1.25 * true_positive / (0.25 * len(actual) + predicted_count)
    return total / len(rows) if rows else 0.0


def best_threshold(rows, grid, per_source: bool) -> tuple[float, float, float]:
    best = (-1.0, 0.5, 0.5)
    if per_source:
        for t2 in grid:
            for t3 in grid:
                value = score_rows(rows, float(t2), float(t3))
                if value > best[0]:
                    best = (value, float(t2), float(t3))
    else:
        for threshold in grid:
            value = score_rows(rows, float(threshold), float(threshold))
            if value > best[0]:
                best = (value, float(threshold), float(threshold))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--train-dir", type=Path, default=root / "dataset" / "train")
    parser.add_argument("--work-dir", type=Path, default=root / "work")
    parser.add_argument("--model-dir", type=Path, default=root / "work" / "model_cap250")
    parser.add_argument("--train-entities", type=int, default=20000)
    parser.add_argument("--candidate-limit", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    targets = [args.train_dir / "train_source2.tsv", args.train_dir / "train_source3.tsv"]
    db = open_index(args.work_dir / "cap250" / "train_index.sqlite", targets, False)
    model = XGBClassifier()
    model.load_model(str(args.model_dir / "model.json"))
    sampled = sample_source1(args.train_dir / "train_source1.tsv", args.train_entities, args.seed)
    selected = {row["entity_id"] for row in sampled}
    truth = load_ground_truth(args.train_dir / "train_ground_truth.tsv", selected)
    validation = []
    covered = possible = 0
    start = time.time()
    try:
        heldout = [
            row for row in sampled
            if int(hashlib.blake2b(row["entity_id"].encode(), digest_size=2).hexdigest(), 16) % 5 == 0
        ]
        for i, row in enumerate(heldout, 1):
            actual = truth.get(row["entity_id"], set())
            candidates = retrieve(db, row, args.candidate_limit)
            possible += len(actual)
            covered += len(actual & {candidate[0] for candidate in candidates})
            context = prepare_entity(row)
            features = np.asarray([pair_features(context, c) for c in candidates], dtype=np.float32)
            probabilities = model.predict_proba(features)[:, 1] if len(features) else np.empty(0)
            validation.append((
                actual,
                [c[0] for c in candidates],
                np.asarray([c[0].startswith("S2-") for c in candidates], dtype=bool),
                probabilities,
                row["entity_id"],
            ))
            if i % 500 == 0:
                print(f"Scored {i:,}/{len(heldout):,} held-out entities", flush=True)
    finally:
        db.close()

    grid_global = np.arange(0.20, 0.901, 0.01)
    grid_source = np.arange(0.20, 0.901, 0.05)
    packed = [row[:4] for row in validation]
    global_best = best_threshold(packed, grid_global, False)
    source_best = best_threshold(packed, grid_source, True)
    current = score_rows(packed, 0.52, 0.52)
    fixed_source_thresholds = (0.50, 0.70)

    cv = {}
    for policy, grid, per_source in (("global", grid_global, False), ("per_source", grid_source, True)):
        fold_results = []
        for tune_fold in (0, 1):
            tune = [
                r[:4] for r in validation
                if int(hashlib.blake2b(r[4].encode(), digest_size=2).hexdigest(), 16) % 2 == tune_fold
            ]
            test = [
                r[:4] for r in validation
                if int(hashlib.blake2b(r[4].encode(), digest_size=2).hexdigest(), 16) % 2 != tune_fold
            ]
            chosen = best_threshold(tune, grid, per_source)
            fold_results.append({
                "tune_thresholds": [chosen[1], chosen[2]],
                "heldout_f05": score_rows(test, chosen[1], chosen[2]),
                "heldout_entities": len(test),
            })
        cv[policy] = fold_results

    fixed_source_cv = []
    for tune_fold in (0, 1):
        test = [
            r[:4] for r in validation
            if int(hashlib.blake2b(r[4].encode(), digest_size=2).hexdigest(), 16) % 2 != tune_fold
        ]
        fixed_source_cv.append({
            "heldout_f05": score_rows(test, *fixed_source_thresholds),
            "heldout_entities": len(test),
        })

    print(json.dumps({
        "validation_entities": len(validation),
        "candidate_recall": covered / possible if possible else 0.0,
        "existing_threshold_0.52_f05": current,
        "best_global_on_full_validation": {
            "f05": global_best[0], "threshold": global_best[1],
        },
        "best_per_source_on_full_validation": {
            "f05": source_best[0], "s2_threshold": source_best[1], "s3_threshold": source_best[2],
        },
        "fixed_source_thresholds": {
            "s2_threshold": fixed_source_thresholds[0],
            "s3_threshold": fixed_source_thresholds[1],
            "full_validation_f05": score_rows(packed, *fixed_source_thresholds),
            "cross_validation": fixed_source_cv,
        },
        "cross_validation": cv,
        "seconds": round(time.time() - start, 1),
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
