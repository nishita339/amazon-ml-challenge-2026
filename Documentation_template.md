# ML Challenge 2026: Business Entity Resolution

**Team Name:** 404 Not Found  
**Team Members:** Nishita Singh, Monika  
**Submission Date:** 2026-09-27

## 1. Executive Summary

This solution retrieves Source 2 and Source 3 records for each Source 1 entity with a country-restricted SQLite FTS5 index, then scores the retrieved pairs with an XGBoost classifier. The threshold is selected on a held-out sample using the challenge's entity-level macro F0.5 metric, including empty-match entities.

The v3 experiment improves local held-out macro F0.5 from 0.8087 to 0.8141 compared with the previous version. This is a local validation result, not a leaderboard score or a guarantee of winning. No external business lookup or enrichment was used.

## 2. Methodology

### 2.1 Data and Constraints

Each Source 1 entity may have zero, one, or multiple matching records across Source 2 and Source 3. Country values are handled dynamically; no country list is hard-coded. Records from different countries are never compared. Text normalization uses Unicode NFKD and case folding, and the process uses only the provided challenge data.

The evaluation is macro-averaged per Source 1 entity with beta = 0.5. Correctly predicting an empty list for a singleton receives full credit; an incorrect non-empty list for a singleton receives zero for that entity.

### 2.2 Pipeline

1. Stream Source 2 and Source 3 into a disk-backed SQLite FTS5 index with normalized business name, address, country, and entity ID.
2. For each Source 1 row, retrieve candidates through country-restricted name phrase/token queries, name-plus-address queries, address-token conjunctions, and three individual informative address-token queries.
3. Stop retrieval at a maximum of 100 candidates per Source 1 entity.
4. Compute name, address, digit, character-trigram, token, and source features for each pair.
5. Score pairs with an XGBoost binary classifier and emit IDs whose probability meets the validation-selected threshold.

The blocker uses a disk-backed full-text index to avoid loading all target records into an in-memory dataframe. The test index contains 9,969,589 Source 2/3 records.

## 3. Candidate Generation

The candidate limit is 100 per Source 1 entity. Candidate recall was measured on the sampled training entities by comparing retrieved IDs with labeled target IDs.

- Previous version candidate recall: 0.7644.
- v3 candidate recall: 0.7761.
- Training sample: 20,000 Source 1 entities, seed 2026.
- The candidate recall figure is a sample estimate; it is not test-set recall because test labels are unavailable.

A separate 1,500-entity miss audit found that many omitted labeled pairs still shared address tokens, indicating that blocking remains the principal recall limitation. Candidate capping and query budgets trade retrieval coverage against candidate-set size and runtime.

## 4. Matching Model

The pair classifier is XGBoost with 401 trees. Features include name token Jaccard/containment, character-trigram Jaccard, address token Jaccard/containment, digit overlap, exact normalized name/address flags, name-length ratio, source indicator, first/last token overlap, and RapidFuzz name/address ratios.

The threshold was selected from a deterministic held-out split using macro F0.5 per Source 1 entity, with the singleton convention included.

- Validation entities: 4,015.
- Selected threshold: 0.62.
- Local held-out macro F0.5: 0.8141.
- Previous version local held-out macro F0.5: 0.8087.

These scores describe the sampled local validation only. The test set has no labels, and the v3 output has not yet received a leaderboard evaluation.

## 5. Test Output and Validation

The v3 prediction processed all 1,732,544 test Source 1 entities:

- `matching_results.tsv`: 1,732,544 rows; 238,192 empty lists and 1,494,352 non-empty lists.
- `candidate_pairs.tsv`: 1,732,544 rows; candidate lists are capped at 100 IDs per Source 1 entity.
- Official validator result: PASS; row counts, headers, duplicate/self-match checks, and matching-subset-of-candidates checks passed.
- The validator's optional full target-ID existence check was not run; target IDs were retrieved directly from the test-set index.

The previous version scored 0.778 on the leaderboard. The v3 output must be submitted separately to learn its leaderboard score; local validation is not a substitute for that evaluation.

## 6. Limitations and Error Analysis

Candidate recall is 77.61% on the training sample, so some true targets never reach the classifier. In the separate miss audit, among missed labeled pairs, 91.4% had at least one normalized address token in common and 79.2% shared a numeric address token. This suggests that retrieval budgets and noisy tokenization still hide recoverable matches. The audit is a diagnostic sample, not a complete manual review.

False positives remain possible when businesses share generic names or locations. The threshold favors precision because false merges are penalized more heavily, but no threshold can guarantee perfect entity-level lists on unseen records.

## Appendix: Reproduction

The runnable implementation is `code/business_entity_resolution/src/run.py`. Model metadata and the trained model are under `code/business_entity_resolution/model/`; pinned dependencies are in `code/business_entity_resolution/requirements.txt`. End-to-end commands are documented in `code/business_entity_resolution/README.md`.

The solution uses no external lookups, private registries, or data outside the challenge files.
