# Business Entity Resolution: ML Challenge 2026

This package contains the runnable SQLite FTS5 + XGBoost pipeline used for the v3 experiment. It uses only the supplied challenge data; it does not call external business registries or lookup services.

## Requirements

Python 3.10+ and the pinned packages in `requirements.txt` are required. Run commands from the `student_resource/` directory.

```powershell
python -m pip install -r code/business_entity_resolution/requirements.txt
```

## Reproduce Training

The checked-in model was trained from a deterministic 20,000-entity sample, with a 20% entity holdout, seed 2026, candidate limit 100, and 401 XGBoost trees.

```powershell
python code/business_entity_resolution/src/run.py train `
  --train-entities 20000 --candidate-limit 100 --seed 2026 --trees 401 `
  --work-dir work --model-dir code/business_entity_resolution/model
```

Training reports the held-out macro F0.5, candidate recall estimate, and selected probability threshold. The recorded local values are 0.8141, 0.7761, and 0.62 respectively; they are not leaderboard results.

## Generate Predictions

```powershell
python code/business_entity_resolution/src/run.py predict `
  --work-dir work --model-dir code/business_entity_resolution/model `
  --output-dir output --candidate-limit 100 --workers 8
```

If a prediction run is interrupted after its output files have been written, resume it with the same paths and add `--resume`. Resume checks the headers and reconciles unpaired partial rows before continuing. Use `--rebuild-index` if the dataset files or index normalizer have changed.

## Validate

```powershell
python utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir dataset/test
```

The optional `--check-ids` flag verifies target IDs against the complete test sources and requires additional memory. The ordinary validator checks row counts, headers, duplicate IDs, self-matches, and that every predicted match is included in the candidate list.

## Implementation Notes

- Candidate retrieval is partitioned by country and capped at 100 IDs per Source 1 row.
- Source 2 and Source 3 are indexed on disk in SQLite FTS5 to limit memory use.
- The classifier uses name, address, number, character-trigram, and source features.
- `src/analyze_blocking_misses.py` audits labeled training matches omitted by blocking.
- The reported local validation score is a sample estimate. The test set is unlabeled, so only the competition portal can report the test score.
