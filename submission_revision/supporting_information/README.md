# Supporting analyses and public audit tables

This supporting package accompanies the Expert Systems manuscript.

## Additional stored-forecast audit, 30 September 2026

`revision_audit.py audit` verifies fixed and nested metrics against their
stored predictions and prepared target records. It writes
`criticism_3_audit/class_support.csv`, `recomputed_metrics.csv`,
`target_subperiod_metrics.csv`, `leave_one_out_details.csv`,
`leave_one_out_summary.csv`, `market_lsa_changed_origins.csv`, and a hash
manifest. The 148 omission calculations retain all fitted models and omit
one common evaluation origin from each of four paired comparisons. Their
minimum/maximum ranges are not confidence intervals. Calendar partitions
use target publication dates and are retrospective descriptive groups,
not forecast inputs, model selection, structural-break tests, or new evidence.
Subperiod class support differs; compare paired models within a period,
not raw macro-F1 across periods. No new significance tests are performed.

Run from the project workspace root:

```bash
python Expert_Systems/supporting_information/revision_audit.py audit
python -m unittest discover -s Expert_Systems/supporting_information -p 'test_revision_audit.py'
```

The audit uses NumPy and the Python standard library, not a GPU or language
model. Supply `--prepared` and `--fixed` to locate the privately retained
`tcmb_next_meeting_prepare_v1.zip` and `publication_run_results.zip` archives
if the `prism-uploads/` layout is unavailable. These raw inputs are not
redistributed in the addendum. The input hashes are recorded in the manifest.

`revision_audit.py prepare-annotator` prepares a separate blinded packet;
`revision_audit.py agreement --answers FILE.csv` computes axis-level raw
agreement, nominal Cohen's kappa, secondary linearly weighted kappa, and
confusion matrices after independent answers are returned. The submitted
second-reference labels have now been compared on all 24
test documents. `second_expert_audit/` contains the agreement table,
confusion matrices, de-identified test labels, integrity records, and
input hashes. Forward-policy nominal kappa is 0.0602; these results do not
establish strong annotation reliability or independent model validation.
The completed `tokenfix` run has now been exported and all 672 response
records have been verified. `rescore_semantic_predictions.py` reproduces
the original strict-format report, 11 published first-object macro-F1
values, and four paired contrast intervals, then evaluates the same frozen
responses against the second reference. `second_expert_rescoring/` supplies
full-reference and common-axis-support metrics, axis results, paired
intervals, 672 parsed responses, the 2,000 shared bootstrap draws, and a
hash manifest. No inference, model refitting, label voting, or adjudication
occurs. Original labels and primary model tables remain unchanged.
The historical agreement-only report describes model rescoring as pending;
the subsequent rescoring manifest records its completion separately.
`export_semantic_predictions.py` remains the CPU-only exporter/discovery
tool for retrieving saved runs. The private identity key and disagreement files must not be
included in the journal's Supporting Information.

Run from the workspace root with the uploaded frozen archive available:

```bash
python Expert_Systems/supporting_information/rescore_semantic_predictions.py
python -m unittest discover -s Expert_Systems/supporting_information -p 'test_*.py'
```

Outside this layout, supply `--archive`, `--original`, `--second`, and
`--output`. The rescoring script uses NumPy and the standard library.
Second-reference intervals are exploratory and conditional on fixed
responses from the same reused 24 test documents; they do not establish
an independent test sample or identify a correct annotator.

## Candidate-package integrity

The current author confirmations permit research and supplementary use of
both non-author annotators' de-identified labels; no consensus labels were
created. The candidate builder includes only explicitly allowlisted public
files and supplies the actual BERT cache from the retained archived results.
It verifies all 122 document/text identities, the 122-by-768 matrix hash,
and pinned encoder commit before inclusion. `PACKAGE_MANIFEST.json` in the
generated ZIP records the public-file inventory and cache verification.
Private keys, raw source archives, adapter weights and adjudication working
files are not included. Preparing these candidates is not a new GitHub or
Zenodo release. AI-tool identification and institutional ethics applicability
remain in the author-facing handoff, not as invented clearance statements.

## Existing analyses

- `regularization_sensitivity.py` reruns the history, history+TF--IDF and
  history+LSA arms for lambda values 0.001, 0.01, 0.1, 1 and 10.
- `regularization_sensitivity.csv` contains all 30 resulting rows.
- `paired_secondary_checks.csv` records the conditional market-proxy MAE
  intervals and exact McNemar direction checks reported in the revision.
- `nested_tuning_analysis.py` performs post-hoc nested rolling-origin selection
  without using the current outer target.
- `nested_tuning_metrics.csv`, `nested_tuning_contrasts.csv`,
  `nested_tuning_choices.csv`, `nested_tuning_inner_scores.csv`,
  `nested_tuning_predictions.csv`, and `nested_tuning_metadata.json` report
  the complete wider-grid tuned analysis, including frozen BERT and all six
  Qwen compositions.
- `nested_selection_stability.csv` reports the selected lambda and scaling
  distribution for direction, magnitude, and probability at every outer
  origin.
- `nested_tuning_probability_metrics.csv`,
  `nested_tuning_probability_predictions.csv`, and
  `nested_probability_selection_comparison.csv` report the variant selected
  by inner multiclass Brier score, with log loss as a tie-breaker.
- `nested_direction_multiplicity.csv`, `nested_white_reality_check.json`, and
  `nested_market_lsa_vs_market.csv` contain the multiplicity audit and direct
  incremental text comparison.
- `bert_embeddings.npz` is the exact-match 122-by-768 frozen BERTurk cache;
  `bert_embedding_validation.json` records the matching encoder commit,
  document identities, and matrix SHA-256.
- `verify_bert_embeddings.py` validates the cache before it is accepted by
  `nested_tuning_analysis.py`.
- `tcmb_statement_source_index.csv` lists meeting dates, official decision
  URLs, split membership, and text hashes.
- `finance_expert_labels.csv` contains the de-identified three-axis reference
  labels; it contains no annotator profile or personal information.
- `qwen_features_public.csv` contains validated archived Qwen feature outputs.
- `public_audit_metadata.json` records source hashes and disclosure checks.
- `row_only` retains the main L2 document normalisation.
- `column_standardized` additionally standardises TF--IDF or LSA columns
  inside each chronological training fold.

The script is designed to run from the public `HawkishvsDovish` repository
and requires the four hash-verified raw archives documented there. This is a
post-hoc robustness analysis and is not used to redefine the retrospective
primary comparison. The regenerated BERT cache matches the matrix SHA-256
from the archived fixed-specification experiment exactly.