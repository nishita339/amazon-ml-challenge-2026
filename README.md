# 🏆 Amazon ML Challenge 2026 — Business Entity Resolution

> **Score: 0.790 F₀.₅** | Ranked among top teams out of 89,000+ registered students  
> A scalable, precision-optimized ML pipeline for resolving business entities across 11.7 million noisy records from 3 independent data sources.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Challenge Overview

The **Amazon ML Challenge 2026** is a 72-hour hackathon where teams build ML solutions for **Business Entity Resolution** — determining which records across multiple noisy data sources refer to the same real-world business.

Given business records from 3 independent sources with noisy, inconsistent fields (names, addresses, countries), the task is to determine which records refer to the same business entity. The evaluation metric is **Macro-Averaged F₀.₅** (precision-heavy: false merges are penalized 2× more than missed matches).

### Key Challenges
- **Scale**: 1.73M reference entities × 9.97M candidate records = potentially 17 trillion comparisons
- **Noise**: Typos, abbreviations, word transpositions, DBA/trade names, non-Latin script transliterations (Hindi, Tamil, Odia, Kannada, Gujarati, French)
- **Unseen Data**: Test set includes France (15% of entities) — completely absent from training data
- **Precision Penalty**: F₀.₅ weights precision 2× over recall

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Country-Partitioned Processing              │
│         (US: 38% | India: 47% | France: 15%)            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. PREPROCESSING                                        │
│     ├── Unicode NFKD normalization (diacritics → ASCII)  │
│     ├── Legal suffix stripping (LLC, Pvt Ltd, SARL...)   │
│     └── Address tokenization & digit extraction          │
│                                                          │
│  2. BLOCKING (Candidate Generation)                      │
│     ├── Canonical clean name index                       │
│     ├── Rare token inverted index (IDF-filtered)         │
│     ├── Address digit + locality concordance index       │
│     └── Posting list pruning (≤50 per key)               │
│     Result: ~5.6 candidates per entity (99.99% reduced)  │
│                                                          │
│  3. MATCHING (Precision-Heavy Decision Engine)           │
│     ├── Rule 1: Exact canonical name + address overlap   │
│     ├── Rule 2: Token Jaccard ≥ 0.5 + address agreement │
│     ├── Rule 3: Domain/hashtag substring concordance     │
│     ├── Rule 4: Deep address match (DBA/transliteration) │
│     └── Rule 5: High name similarity fallback            │
│     Optimized for: Macro F₀.₅ > 0.98 on validation      │
│                                                          │
│  4. OUTPUT                                               │
│     ├── matching_results.tsv (1,732,544 rows)            │
│     └── candidate_pairs.tsv  (1,732,544 rows)            │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Results

| Metric | Value |
|:---|:---|
| **Leaderboard F₀.₅ Score** | **0.790** |
| **Local Validation Precision** | 99.83% |
| **Local Validation Recall** | 95.54% |
| **Blocking Recall Ceiling** | 96.84% |
| **Avg Candidates per Entity** | 5.64 |
| **Total Entities Processed** | 1,732,544 |
| **Total Records Indexed** | 9,969,589 |

---

## 📁 Project Structure

```
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py          # Package exports
│       │   ├── preprocess.py        # Unicode normalization, legal term & stopword stripping
│       │   ├── blocking.py          # Inverted index blocking & candidate generation
│       │   ├── matcher.py           # Pairwise scoring & precision-weighted match rules
│       │   └── run_pipeline.py      # End-to-end pipeline runner
│       ├── model/                   # Trained XGBoost model artifacts
│       ├── README.md                # Reproduction guide
│       └── requirements.txt         # Dependencies
├── utils/
│   └── validate_submission.py       # Official competition validator
├── Documentation_template.md        # Technical methodology report
├── README.md                        # This file
└── .gitignore
```

> **Note**: The `dataset/` and `output/` directories are excluded from git due to size (~2GB+). Download the dataset from the [competition page](https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604).

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/<your-username>/amazon-ml-challenge-2026.git
cd amazon-ml-challenge-2026
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Download Dataset
Download the dataset from the competition page and extract it into the `dataset/` directory:
```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### 3. Run the Pipeline
```bash
python code/business_entity_resolution/src/run_pipeline.py \
    --test-dir dataset/test \
    --output-dir output \
    --max-candidates 8
```

### 4. Validate Output
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 🔬 Key Technical Insights

### 1. Zero Cross-Country Matches
Empirical analysis of 34,752 ground truth pairs revealed that **0% of matches cross country boundaries**. Partitioning by country reduces the search space by ~65% with zero recall loss.

### 2. Noise Taxonomy (Observed Patterns)
| Pattern | Example |
|:---|:---|
| **Legal Suffix Drift** | `BS Projects` → `BS Projects Ltd Ltd` |
| **Word Transposition** | `Physical Therapy Associates Group` → `Group Physical Therapy Asseatiaes` |
| **DBA / Trade Names** | `Probst & Duran Newhold LLC` → `NYLADREX` (same address) |
| **Domain Names** | `SJ Ace Vendome Inc` → `sjacevendome.com` / `#sjace` |
| **Non-Latin Transliteration** | `Shakti Agro Limited` → `ଶକ୍ତି ଆଗ୍ରୋ ଲିମିଟେଡ୍` (Odia) |
| **Address Abbreviation** | `Uttar Pradesh` → `UP` / `उत्तर प्रदेश` |
| **Diacritics** | `Novent Owl PLLC` → `PLLC Novent Ówl` |

### 3. Precision vs Recall Trade-off
F₀.₅ weights precision **2× over recall**. A single false merge costs twice as much as a missed match. Our matching rules are deliberately conservative — preferring to miss a hard-to-confirm match rather than risk a false merge.

### 4. Singleton Handling
**5.58%** of entities have zero matches. Correctly predicting an empty list earns a perfect 1.0 for that entity; any false match on a singleton earns 0.0. Our pipeline correctly identifies **100,455 singletons** in the test set.

---

## 📈 Evaluation Formula

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

Computed as a **macro-average**: F₀.₅ is calculated per Source 1 entity, then averaged across all entities.

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Core Libraries**: Standard library only (re, unicodedata, collections) for the main pipeline
- **ML Framework**: XGBoost (for experimental classifier variants)
- **Validation**: scikit-learn
- **No External APIs**: All processing uses only the provided dataset (competition requirement)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Amazon** for organizing the ML Challenge 2026
- **AWS** for providing $200 in free credits to all participants
- **Unstop** for hosting the competition platform
