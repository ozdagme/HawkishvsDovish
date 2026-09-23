# TCMB decision-text forecasting: reproducibility repository

This repository reproduces the chronological forecasting comparison and runs the two additional experiments requested during journal preparation:

1. a frozen contextual-encoder baseline (`history_transformer`); and
2. Qwen feature ablations by economic axis and missingness component.

The pipeline makes **no new Qwen calls**. It uses the archived, validated Qwen outputs distributed in `data/raw/tcmb_qwen_features_v1.zip`.

## Scientific status

The additional experiments use the same 37 forecast origins that were already inspected during manuscript development. They are therefore a **post-hoc robustness extension**, not a new independent confirmatory test. The pipeline records this status in every publication run.

The repository deliberately retains one finance-expert reference set. It does not claim inter-rater reliability or expert consensus.

## Repository layout

```text
configs/publication.json       Frozen experiment settings
data/raw/                      Hash-pinned source archives
data/derived/                  Generated embedding caches
docs/                          Protocol and troubleshooting notes
outputs/                       Generated predictions, metrics and manifests
src/tcmb_repro/pipeline.py     Scientific pipeline
tests/                         Unit and integration checks
run.py                         Command-line entry point
```

## 1. System requirements

- Python 3.10–3.12 is recommended for the transformer run.
- CPU execution is supported. A CUDA GPU shortens embedding extraction.
- Approximately 4 GB free disk space is recommended for the Python environment and model cache.
- Git LFS is not required for the files currently included in this repository.

## 2. Installation

Create and activate an environment using your preferred environment manager. Then install a PyTorch build appropriate for your operating system and, if applicable, CUDA version. Finally run:

```bash
python -m pip install -r requirements.txt
```

For a CPU-only installation, consult the current PyTorch installation selector before installing `requirements.txt`; PyTorch wheel addresses vary by platform.

## 3. Verify the repository

Place the verified source snapshot at `data/raw/data.zip`. Then stage the
remaining three archives from the existing experiment outputs:

```bash
python run.py stage-data --source-dir /path/to/HawkishvsDovish/outputs
```

The command searches both direct ZIP files and ZIP members nested inside
archived output bundles. An already present `data/raw/data.zip` is retained
after its SHA-256 identity is verified. All four frozen archives are required,
and only files with the expected SHA-256 identities are accepted.

Then run:

```bash
python run.py check
```

Expected invariants:

- 122 origins with observed next-meeting targets;
- 37 chronological evaluation folds;
- first evaluation origin: 19 January 2023;
- last evaluation origin: 11 June 2026;
- 95 valid and 27 invalid archived Qwen feature responses.

The command stops if any source archive has a different SHA-256 hash.

## 4. Run the software smoke test

```bash
python run.py smoke --force
```

This command uses deterministic 64-dimensional hash vectors to test every fold, model, metric and output writer without downloading a transformer. Outputs go to `outputs/smoke/` and contain `SMOKE_ONLY.json`.

**Never report the hash-embedding results in the paper.** They are not a contextual-language-model baseline.

## 5. Run the publication experiment

### One command

```bash
python run.py all
```

This command:

1. verifies all input hashes;
2. downloads/loads the frozen Turkish BERT encoder configured in `configs/publication.json`;
3. encodes all decision statements using overlapping token chunks;
4. stores the fixed document embeddings in `data/derived/bert_embeddings.npz`;
5. reruns all 37 expanding-window forecasting folds;
6. fits the contextual baseline and all Qwen ablations;
7. computes paired circular-block intervals using common bootstrap indices;
8. writes predictions, metrics, fold audits and file hashes.

### Two-stage execution

Embedding extraction and forecasting can be run separately:

```bash
python run.py embed
python run.py run
```

Use `--force` only when intentionally replacing an existing cache or output directory.

## 6. Outputs used for the manuscript

The publication run writes:

- `outputs/publication_run/predictions.csv`
- `outputs/publication_run/metrics.csv`
- `outputs/publication_run/contrasts_vs_history.csv`
- `outputs/publication_run/fold_audit.json`
- `outputs/publication_run/run_metadata.json`
- `outputs/publication_run/manifest.json`
- `outputs/publication_run/RESULTS.md`

Before copying numbers into the manuscript, confirm that `run_metadata.json` contains:

```json
"backend": "transformer"
```

and that `resolved_commit` is not empty. The resolved Hugging Face model commit should be copied into the final methods and reproducibility statements.

## 7. New model arms

| Arm | Description |
|---|---|
| `history_transformer` | Decision-history baseline plus frozen Turkish BERT document embedding |
| `history_qwen_inflation` | Baseline plus inflation-pressure score and its missingness indicator |
| `history_qwen_demand` | Baseline plus demand-pressure score and its missingness indicator |
| `history_qwen_forward` | Baseline plus forward-policy-bias score and its missingness indicator |
| `history_qwen_scores_only` | Baseline plus all three scores, excluding missingness indicators |
| `history_qwen_missing_only` | Baseline plus the three missingness indicators, excluding scores |
| `history_qwen_all` | Baseline plus all three scores and all three missingness indicators |

All numeric scaling, median imputation and predictive-model estimation are learned only from the current fold’s training observations.

## 8. Tests

Run:

```bash
python -m unittest discover -s tests -v
```

The integration test invokes the complete smoke pipeline in a temporary output directory.

## 9. Reproducibility rules

- Do not edit files under `data/raw/`.
- Do not report results from a changed configuration under the original protocol name.
- Do not interpret the transformer comparison as independent confirmation.
- Do not describe a confidence interval containing zero as evidence of superiority.
- Do not upload model caches or access tokens to GitHub.
- Preserve `manifest.json` with every archived result package.

## Citation

Use `CITATION.cff` after filling the repository release URL and DOI.