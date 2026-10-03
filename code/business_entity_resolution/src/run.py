#!/usr/bin/env python3
"""Disk-backed business entity resolution pipeline for ML Challenge 2026."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
from itertools import zip_longest
import json
import random
import re
import sqlite3
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
from rapidfuzz import fuzz
from xgboost import XGBClassifier


NORMALIZER_VERSION = 3
LEGAL_SUFFIXES = {
    "llc", "llp", "pllc", "inc", "incorporated", "corp", "corporation",
    "ltd", "limited", "company", "co", "private", "pvt", "sarl", "sas",
    "sasu", "sa", "eurl", "snc", "gmbh", "llc.",
}
ADDRESS_WORDS = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "hwy": "highway",
    "ln": "lane", "dr": "drive", "ste": "suite", "apt": "apartment",
    "fl": "floor", "pkwy": "parkway",
}
ADDRESS_STOP = {
    "street", "road", "avenue", "boulevard", "highway", "lane", "drive",
    "suite", "apartment", "floor", "near", "opposite", "india", "united",
    "states", "county", "district", "city", "block", "sector", "building",
    "house", "number", "plot", "north", "south", "east", "west",
}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
DIGIT_RE = re.compile(r"\d+", re.UNICODE)
THREAD_STATE = threading.local()


def normalize(value: str, address: bool = False) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch)).casefold()
    tokens = TOKEN_RE.findall(value)
    if address:
        tokens = [ADDRESS_WORDS.get(token, token) for token in tokens]
    else:
        while tokens and tokens[-1].rstrip(".") in LEGAL_SUFFIXES:
            tokens.pop()
        if len(tokens) > 1 and tokens[-2:] in (["private", "limited"], ["pvt", "ltd"]):
            tokens = tokens[:-2]
    return " ".join(tokens)


def tokens(value: str, min_length: int = 1) -> set[str]:
    return {t for t in TOKEN_RE.findall(value) if len(t) >= min_length}


def fts_term_group(field: str, terms: list[str]) -> str:
    return field + " : (" + " AND ".join('"' + t.replace('"', '""') + '"' for t in terms) + ")"


def file_signature(paths: Iterable[Path]) -> str:
    values = []
    for path in paths:
        stat = path.stat()
        values.append((path.name, stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(values).encode("utf-8")).hexdigest()


def open_index(path: Path, target_paths: list[Path], rebuild: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists() and path.stat().st_size > 0
    db = sqlite3.connect(str(path))
    db.execute("PRAGMA cache_size=-64000")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA mmap_size=0")
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    if existed and not rebuild:
        try:
            meta = dict(db.execute("SELECT key, value FROM metadata"))
        except sqlite3.Error as exc:
            db.close()
            raise RuntimeError(f"Index exists but is unreadable: {path}; use --rebuild-index") from exc
        expected = file_signature(target_paths)
        if meta.get("signature") != expected or meta.get("normalizer") != str(NORMALIZER_VERSION):
            db.close()
            raise RuntimeError(f"Index does not match the selected data: {path}; use --rebuild-index")
        return db

    db.execute("DROP TABLE IF EXISTS records")
    db.execute("DROP TABLE IF EXISTS metadata")
    db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("""CREATE VIRTUAL TABLE records USING fts5(
        entity_id UNINDEXED, country, name, address,
        tokenize='unicode61 remove_diacritics 2'
    )""")
    start = time.time()
    total = 0
    batch = []
    with ExitStack() as stack:
        readers = []
        for source_path in target_paths:
            handle = stack.enter_context(source_path.open("r", encoding="utf-8-sig", newline=""))
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"entity_id", "business_name", "business_address", "country"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected columns in {source_path}")
            readers.append(reader)
        for rows in zip_longest(*readers):
            for row in rows:
                if row is None:
                    continue
                batch.append((
                    row["entity_id"], row["country"],
                    normalize(row["business_name"]), normalize(row["business_address"], True),
                ))
                if len(batch) >= 2000:
                    db.executemany("INSERT INTO records VALUES (?, ?, ?, ?)", batch)
                    total += len(batch)
                    batch.clear()
                    db.commit()
                    if total % 200000 == 0:
                        print(f"Indexed {total:,} records ({time.time() - start:.0f}s)", flush=True)
    if batch:
        db.executemany("INSERT INTO records VALUES (?, ?, ?, ?)", batch)
        total += len(batch)
        db.commit()
    db.executemany("INSERT INTO metadata VALUES (?, ?)", [
        ("signature", file_signature(target_paths)),
        ("normalizer", str(NORMALIZER_VERSION)),
        ("record_count", str(total)),
    ])
    db.commit()
    print(f"Built index with {total:,} records in {time.time() - start:.0f}s: {path}", flush=True)
    return db


def retrieve(db: sqlite3.Connection, row: dict[str, str], limit: int) -> list[tuple[str, str, str]]:
    name = normalize(row.get("business_name", ""))
    address = normalize(row.get("business_address", ""), True)
    name_terms = sorted(tokens(name, 2), key=lambda t: (len(t), t), reverse=True)
    address_terms = sorted(
        (t for t in tokens(address, 3) if t not in ADDRESS_STOP),
        key=lambda t: (len(t), t), reverse=True,
    )
    country = '"' + row.get("country", "").replace('"', '""') + '"'
    queries: list[tuple[str, int]] = []
    if name_terms:
        phrase_terms = list(dict.fromkeys(name.split()))[:6]
        if phrase_terms:
            phrase = '"' + " ".join(phrase_terms) + '"'
            queries.append(("country : " + country + " AND name : " + phrase, max(8, limit // 5)))
        if len(name_terms) >= 2:
            queries.append(("country : " + country + " AND " + fts_term_group("name", name_terms[:3]), max(10, limit // 4)))
    if name_terms and address_terms:
        queries.append((
            "country : " + country + " AND "
            + fts_term_group("name", name_terms[:2]) + " AND "
            + fts_term_group("address", address_terms[:2]), max(10, limit // 4)
        ))
    if len(address_terms) >= 2:
        queries.append(("country : " + country + " AND " + fts_term_group("address", address_terms[:2]), max(10, limit // 4)))
    elif address_terms:
        queries.append(("country : " + country + " AND " + fts_term_group("address", address_terms[:1]), max(10, limit // 4)))
    for term in address_terms[:3]:
        queries.append((
            "country : " + country + " AND " + fts_term_group("address", [term]),
            max(4, limit // 40),
        ))
    if name_terms:
        queries.append(("country : " + country + " AND " + fts_term_group("name", name_terms[:1]), max(8, limit // 6)))

    found = {}
    for expression, budget in queries:
        for candidate in db.execute(
            "SELECT entity_id, name, address FROM records "
            "WHERE records MATCH ? LIMIT ?",
            (expression, budget),
        ):
            found.setdefault(candidate[0], candidate)
            if len(found) >= limit:
                return list(found.values())
    return list(found.values())


def retrieve_threaded(task: tuple[dict[str, str], str, int]) -> list[tuple[str, str, str]]:
    row, db_uri, limit = task
    db = getattr(THREAD_STATE, "db", None)
    if db is None:
        db = sqlite3.connect(db_uri, uri=True)
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA cache_size=-16000")
        db.execute("PRAGMA mmap_size=0")
        THREAD_STATE.db = db
    return retrieve(db, row, limit)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def trigrams(value: str) -> set[str]:
    compact = " " + value + " "
    return {compact[i:i + 3] for i in range(max(0, len(compact) - 2))}


def prepare_entity(row: dict[str, str]) -> dict[str, object]:
    name = normalize(row.get("business_name", ""))
    address = normalize(row.get("business_address", ""), True)
    name_parts = name.split()
    return {
        "name": name,
        "address": address,
        "name_tokens": tokens(name, 1),
        "address_tokens": tokens(address, 1),
        "numbers": set(DIGIT_RE.findall(address)),
        "trigrams": trigrams(name),
        "first_token": name_parts[0] if name_parts else "",
        "last_token": name_parts[-1] if name_parts else "",
    }


def pair_features(context: dict[str, object], candidate: tuple[str, str, str]) -> list[float]:
    entity_id, other_name, other_address = candidate
    name = context["name"]
    address = context["address"]
    name_a, name_b = context["name_tokens"], tokens(other_name, 1)
    addr_a, addr_b = context["address_tokens"], tokens(other_address, 1)
    nums_a, nums_b = context["numbers"], set(DIGIT_RE.findall(other_address))
    tri_a, tri_b = context["trigrams"], trigrams(other_name)
    tri_similarity = jaccard(tri_a, tri_b)
    shorter = min(len(name), len(other_name))
    longer = max(len(name), len(other_name))
    shorter_digits = min(len(nums_a), len(nums_b))
    longer_digits = max(len(nums_a), len(nums_b))
    return [
        jaccard(name_a, name_b), containment(name_a, name_b),
        tri_similarity, jaccard(addr_a, addr_b),
        containment(addr_a, addr_b), jaccard(nums_a, nums_b),
        float(bool(name) and name == other_name),
        float(bool(address) and address == other_address),
        shorter / longer if longer else 0.0,
        float(entity_id.startswith("S2-")),
        float(bool(context["first_token"]) and context["first_token"] in name_b),
        fuzz.ratio(name, other_name) / 100.0 if name and other_name else 0.0,
        fuzz.token_sort_ratio(name, other_name) / 100.0 if name and other_name else 0.0,
        fuzz.token_sort_ratio(address, other_address) / 100.0 if address and other_address else 0.0,
        shorter_digits / longer_digits if longer_digits else 0.0,
        float(bool(nums_a & nums_b)),
        float(bool(context["last_token"]) and context["last_token"] in name_b),
    ]


FEATURE_NAMES = [
    "name_jaccard", "name_containment", "name_trigram_jaccard",
    "address_jaccard", "address_containment", "digit_jaccard",
    "exact_name", "exact_address", "name_length_ratio", "is_source2",
    "first_token_overlap", "name_sequence_ratio", "name_sorted_ratio",
    "address_sorted_ratio", "digit_containment", "exact_digit_overlap",
    "last_token_overlap",
]


def load_ground_truth(path: Path, selected: set[str]) -> dict[str, set[str]]:
    truth = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            source_id = row["source1_entity_id"]
            if source_id in selected:
                truth[source_id] = set(filter(None, row["matched_entity_ids"].split(",")))
    return truth


def sample_source1(path: Path, count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    sample = []
    seen = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            seen += 1
            if len(sample) < count:
                sample.append(row)
            else:
                slot = rng.randrange(seen)
                if slot < count:
                    sample[slot] = row
    return sample


def entity_fscore(actual: set[str], predicted: set[str]) -> float:
    if not actual:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    precision = len(actual & predicted) / len(predicted)
    recall = len(actual & predicted) / len(actual)
    denom = 0.25 * precision + recall
    return 1.25 * precision * recall / denom if denom else 0.0


def macro_f05(rows: list[tuple[set[str], list[tuple[str, float]]]], threshold: float) -> float:
    if not rows:
        return 0.0
    return sum(entity_fscore(actual, {entity_id for entity_id, p in scored if p >= threshold})
               for actual, scored in rows) / len(rows)


def train(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)
    targets = [train_dir / "train_source2.tsv", train_dir / "train_source3.tsv"]
    db = open_index(Path(args.work_dir) / "train_index.sqlite", targets, args.rebuild_index)
    try:
        sampled = sample_source1(train_dir / "train_source1.tsv", args.train_entities, args.seed)
        selected = {row["entity_id"] for row in sampled}
        truth = load_ground_truth(train_dir / "train_ground_truth.tsv", selected)
        x_train, y_train = [], []
        validation = []
        covered, possible = 0, 0
        started = time.time()
        for i, row in enumerate(sampled, 1):
            source_id = row["entity_id"]
            actual = truth.get(source_id, set())
            candidates = retrieve(db, row, args.candidate_limit)
            possible += len(actual)
            covered += len(actual & {candidate[0] for candidate in candidates})
            context = prepare_entity(row)
            features = [pair_features(context, candidate) for candidate in candidates]
            labels = [int(candidate[0] in actual) for candidate in candidates]
            is_validation = int(hashlib.blake2b(source_id.encode(), digest_size=2).hexdigest(), 16) % 5 == 0
            if is_validation:
                validation.append((actual, [(candidate[0], feature) for candidate, feature in zip(candidates, features)]))
            else:
                x_train.extend(features)
                y_train.extend(labels)
            if i % 1000 == 0:
                print(f"Prepared {i:,}/{len(sampled):,} training entities ({time.time() - started:.0f}s)", flush=True)
        if not x_train or len(set(y_train)) < 2:
            raise RuntimeError("Training sample did not produce both positive and negative pairs")
        model = XGBClassifier(
            n_estimators=args.trees, max_depth=6, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.9, min_child_weight=2,
            reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=args.threads, random_state=args.seed,
        )
        model.fit(np.asarray(x_train, dtype=np.float32), np.asarray(y_train, dtype=np.int8), verbose=False)
        val_scored = []
        for actual, candidates in validation:
            if candidates:
                probs = model.predict_proba(np.asarray([f for _, f in candidates], dtype=np.float32))[:, 1]
                val_scored.append((actual, [(candidate[0], float(p)) for candidate, p in zip(candidates, probs)]))
            else:
                val_scored.append((actual, []))
        thresholds = [i / 100 for i in range(10, 96, 2)]
        best_threshold = max(thresholds, key=lambda t: macro_f05(val_scored, t))
        score = macro_f05(val_scored, best_threshold)
        args.model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(args.model_dir / "model.json"))
        metadata = {
            "threshold": best_threshold,
            "candidate_limit": args.candidate_limit,
            "features": FEATURE_NAMES,
            "validation_macro_f05": score,
            "validation_entities": len(validation),
            "candidate_recall_sample": covered / possible if possible else 0.0,
            "training_entities_sampled": len(sampled),
            "positive_candidate_pairs": int(sum(y_train)),
            "trainable_candidate_pairs": len(y_train),
            "seed": args.seed,
            "normalizer_version": NORMALIZER_VERSION,
        }
        (args.model_dir / "model_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)
    finally:
        db.close()


def predict(args: argparse.Namespace) -> None:
    model_path = args.model_dir / "model.json"
    metadata_path = args.model_dir / "model_meta.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Trained model is missing; run the train command first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("normalizer_version") != NORMALIZER_VERSION:
        raise RuntimeError("Model was trained with a different normalizer version")
    threshold = args.threshold if args.threshold is not None else float(metadata["threshold"])
    limit = args.candidate_limit or int(metadata["candidate_limit"])
    model = XGBClassifier()
    model.load_model(str(model_path))

    test_dir = Path(args.test_dir)
    targets = [test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"]
    index_path = Path(args.work_dir) / "test_index.sqlite"
    db = open_index(index_path, targets, args.rebuild_index)
    db.close()
    db_uri = index_path.resolve().as_uri() + "?mode=ro"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_pairs.tsv"
    matching_path = args.output_dir / "matching_results.tsv"
    completed = set()
    if args.resume:
        expected_headers = (
            (candidate_path, ["source1_entity_id", "candidate_entity_ids"]),
            (matching_path, ["source1_entity_id", "matched_entity_ids"]),
        )
        completed_ids = []
        for path, expected_header in expected_headers:
            if not path.is_file():
                raise FileNotFoundError(f"Cannot resume; output file is missing: {path}")
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                if next(reader, None) != expected_header:
                    raise ValueError(f"Cannot resume; unexpected header in {path}")
                completed_ids.append({row[0] for row in reader if row})
        if completed_ids[0] != completed_ids[1]:
            completed = completed_ids[0] & completed_ids[1]
            dropped = sum(len(ids - completed) for ids in completed_ids)
            for path, expected_header in expected_headers:
                temporary = path.with_suffix(path.suffix + ".resume")
                with path.open("r", encoding="utf-8", newline="") as source_handle, \
                     temporary.open("w", encoding="utf-8", newline="") as output_handle:
                    reader = csv.reader(source_handle, delimiter="\t")
                    next(reader, None)
                    writer = csv.writer(output_handle, delimiter="\t", lineterminator="\n")
                    writer.writerow(expected_header)
                    for row in reader:
                        if row and row[0] in completed:
                            writer.writerow(row)
                temporary.replace(path)
            print(f"Removed {dropped:,} unpaired partial rows before resuming", flush=True)
        else:
            completed = completed_ids[0]
        print(f"Resuming after {len(completed):,} completed entities", flush=True)
    started = time.time()
    try:
        with (test_dir / "test_source1.tsv").open("r", encoding="utf-8-sig", newline="") as source, \
             candidate_path.open("a" if args.resume else "w", encoding="utf-8", newline="") as candidates_out, \
             matching_path.open("a" if args.resume else "w", encoding="utf-8", newline="") as matching_out:
            reader = csv.DictReader(source, delimiter="\t")
            candidate_writer = csv.writer(candidates_out, delimiter="\t", lineterminator="\n")
            matching_writer = csv.writer(matching_out, delimiter="\t", lineterminator="\n")
            if not args.resume:
                candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
                matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
            batch = []

            def write_batch() -> None:
                if not batch:
                    return
                flat_features = [feature for _, retrieved, features in batch for feature in features]
                probabilities = model.predict_proba(np.asarray(flat_features, dtype=np.float32))[:, 1] if flat_features else []
                offset = 0
                for source_id, retrieved, features in batch:
                    scores = probabilities[offset:offset + len(features)]
                    offset += len(features)
                    matched = [candidate[0] for candidate, probability in zip(retrieved, scores)
                               if probability >= threshold]
                    candidate_writer.writerow([source_id, ",".join(candidate[0] for candidate in retrieved)])
                    matching_writer.writerow([source_id, ",".join(matched)])
                batch.clear()

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = deque()
                rows = iter(reader)

                def enqueue() -> bool:
                    while True:
                        try:
                            row = next(rows)
                        except StopIteration:
                            return False
                        if row["entity_id"] in completed:
                            continue
                        future = pool.submit(retrieve_threaded, (row, db_uri, limit))
                        pending.append((row, future))
                        return True

                for _ in range(args.workers * 8):
                    if not enqueue():
                        break
                i = 0
                while pending:
                    row, future = pending.popleft()
                    retrieved = future.result()
                    enqueue()
                    context = prepare_entity(row)
                    features = [pair_features(context, candidate) for candidate in retrieved]
                    batch.append((row["entity_id"], retrieved, features))
                    i += 1
                    if len(batch) >= 512:
                        write_batch()
                    if i % 10000 == 0:
                        write_batch()
                        candidates_out.flush()
                        matching_out.flush()
                        print(f"Predicted {i:,} entities ({time.time() - started:.0f}s)", flush=True)
            write_batch()
    finally:
        db.close()
    print(f"Wrote {matching_path} and {candidate_path} in {time.time() - started:.0f}s", flush=True)


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    package_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Train and run the entity resolution pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help="build a train index, fit XGBoost, and tune F0.5 threshold")
    train_parser.add_argument("--train-dir", type=Path, default=root / "dataset" / "train")
    train_parser.add_argument("--work-dir", type=Path, default=root / "work")
    train_parser.add_argument("--model-dir", type=Path, default=package_root / "model")
    train_parser.add_argument("--train-entities", type=int, default=8000)
    train_parser.add_argument("--candidate-limit", type=int, default=100)
    train_parser.add_argument("--trees", type=int, default=350)
    train_parser.add_argument("--threads", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--rebuild-index", action="store_true")
    train_parser.set_defaults(func=train)

    predict_parser = sub.add_parser("predict", help="generate test submission files")
    predict_parser.add_argument("--test-dir", type=Path, default=root / "dataset" / "test")
    predict_parser.add_argument("--work-dir", type=Path, default=root / "work")
    predict_parser.add_argument("--model-dir", type=Path, default=package_root / "model")
    predict_parser.add_argument("--output-dir", type=Path, default=root / "output")
    predict_parser.add_argument("--candidate-limit", type=int)
    predict_parser.add_argument("--threshold", type=float)
    predict_parser.add_argument("--workers", type=int, default=4)
    predict_parser.add_argument("--resume", action="store_true", help="resume appending to matching output files")
    predict_parser.add_argument("--rebuild-index", action="store_true")
    predict_parser.set_defaults(func=predict)
    return p


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
