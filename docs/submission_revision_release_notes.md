# Submission revision: frozen-response and second-reference audit

This update adds the 30 September 2026 submission-revision audits to the
previously archived reproducibility baseline. The release tag and DOI must
be assigned from the actual published repository state; this document does
not establish that a new release has already been published.

## Included

- Stored-forecast single-origin omission and descriptive target-date
  calendar audits; no model refitting or new forecast outcomes.
- Independent, non-adjudicated labels for the same 24 semantic test
  documents and pre-adjudication agreement tables. Both annotators are
  non-authors and permitted use of their de-identified labels.
- Frozen semantic-response rescoring against original and second
  references, with full-reference and common-axis-support variants.
- Parsed content for 672 stored responses, model and axis metrics,
  paired conditional intervals, shared bootstrap indices and hash records.
- The actual 122-by-768 BERT cache, matched to the source-document hashes,
  archived matrix SHA-256 and pinned encoder commit.
- CPU-only audit/export/rescoring scripts and regression tests.

## Scientific status

The original chronological splits, labels, fitted models and primary model
tables are preserved. Eleven published first-object F1 values and four
paired intervals reproduce from the frozen run. The original strict-format
report and first-object content scores are validated separately. Under the
second reference, the Qwen fine-tuning interval includes zero and the
no-RAG fine-tuned macro-F1 ranking reverses. Forward-policy inter-annotator
kappa is 0.0602. The reused 24 documents do not provide independent
confirmation, reference adjudication, or strong reliability evidence for
the full 122-document corpus.

Private identity keys, annotator declarations and personal information,
adjudication working files, model weights and raw third-party archives are
excluded. Existing software/data/model licensing terms are not changed.
The archived baseline and its DOI are retained; a DOI for this revision
must not be inferred or copied from the older version.