#!/usr/bin/env python3
"""
End-to-End Scalable Entity Resolution Pipeline for Amazon ML Challenge 2026.

Features:
- Partitioned country streaming (100% memory-safe: France, US, India).
- Multilingual normalization, international legal suffix stripping, and address canonicalization.
- Inverted index blocking with pruned posting lists (low candidate footprint).
- Precision-heavy matching engine optimized for Macro F_0.5.
- Exact row-by-row alignment with test_source1.tsv.
"""

import argparse
import gc
import os
import sys
import time
from collections import defaultdict

# Add parent directory to path to allow direct invocation
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocess import clean_text, get_name_tokens, get_addr_tokens, get_digits
from blocking import BlockingIndex
from matcher import score_and_match

def run_pipeline(test_dir: str, output_dir: str, max_candidates: int = 8):
    os.makedirs(output_dir, exist_ok=True)
    matching_out = os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    for p in [s1_path, s2_path, s3_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required dataset file not found: {p}")

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 - ENTITY RESOLUTION PIPELINE")
    print("=" * 70)
    start_time = time.time()

    # Step 1: Discover countries and order of S1
    print("\n[Step 1/4] Discovering country partitions from test_source1.tsv...")
    country_s1_ids = defaultdict(list)
    s1_order = []

    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                eid = parts[0]
                country = parts[3]
                country_s1_ids[country].append(eid)
                s1_order.append(eid)

    total_s1 = len(s1_order)
    print(f"Discovered {len(country_s1_ids)} country partitions across {total_s1:,} S1 entities:")
    for c, ids in country_s1_ids.items():
        print(f"  - {c}: {len(ids):,} entities ({len(ids)/total_s1*100:.1f}%)")

    # Storage for predictions across countries
    all_candidates = {}
    all_matches = {}

    # Step 2 & 3: Process each country partition
    print("\n[Step 2/4] Processing country partitions (Candidate Generation + Matching)...")
    for country_idx, (country, s1_id_list) in enumerate(country_s1_ids.items(), start=1):
        print(f"\n--- [{country_idx}/{len(country_s1_ids)}] Partition: {country} ({len(s1_id_list):,} S1 entities) ---")
        c_start = time.time()

        # Build Inverted Blocking Index for this country
        index = BlockingIndex(max_postings=50)

        # Index Source 2
        print(f"  Indexing Source 2 ({country})...")
        s2_cnt = 0
        with open(s2_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    index.add_record(parts[0], parts[1], parts[2])
                    s2_cnt += 1

        # Index Source 3
        print(f"  Indexing Source 3 ({country})...")
        s3_cnt = 0
        with open(s3_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    index.add_record(parts[0], parts[1], parts[2])
                    s3_cnt += 1

        print(f"  Total records in {country} target database: {len(index.records_db):,} (S2: {s2_cnt:,}, S3: {s3_cnt:,})")
        index.filter_postings()

        # Stream S1 records for this country and resolve
        print(f"  Resolving entities for {country}...")
        res_cnt = 0
        matches_found = 0
        cands_generated = 0

        # We can scan test_source1 filtering for this country
        with open(s1_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    eid = parts[0]
                    name = parts[1]
                    addr = parts[2]

                    cands = index.generate_candidates(name, addr)
                    top_cands, matched_ids = score_and_match(
                        name, addr, cands, index.records_db, max_candidates=max_candidates
                    )

                    all_candidates[eid] = ",".join(top_cands)
                    all_matches[eid] = ",".join(matched_ids)

                    cands_generated += len(top_cands)
                    matches_found += len(matched_ids)
                    res_cnt += 1

        c_time = time.time() - c_start
        print(f"  Finished {country} in {c_time:.2f}s ({res_cnt/c_time:.1f} entities/sec).")
        print(f"  Avg candidates: {cands_generated/res_cnt:.2f} | Avg matches: {matches_found/res_cnt:.2f}")

        # Free memory
        del index
        gc.collect()

    # Step 4: Write final submission files in exact S1 order
    print("\n[Step 4/4] Writing final submission files with exact sequence alignment...")
    t_write = time.time()

    with open(matching_out, "w", encoding="utf-8") as fm, \
         open(candidate_out, "w", encoding="utf-8") as fc:

        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for eid in s1_order:
            m_str = all_matches.get(eid, "")
            c_str = all_candidates.get(eid, "")
            fm.write(f"{eid}\t{m_str}\n")
            fc.write(f"{eid}\t{c_str}\n")

    print(f"  Wrote {len(s1_order):,} rows to {matching_out} and {candidate_out} in {time.time()-t_write:.2f}s.")
    print("=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {time.time()-start_time:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Amazon ML Challenge 2026 Entity Resolution Pipeline")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test dataset directory")
    parser.add_argument("--output-dir", default="output", help="Path to output directory")
    parser.add_argument("--max-candidates", type=int, default=8, help="Max candidates per S1 entity")
    args = parser.parse_args()

    run_pipeline(args.test_dir, args.output_dir, args.max_candidates)
