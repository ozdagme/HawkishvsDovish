# Experiment protocol

## Research questions

### Contextual-encoder baseline

Does adding a frozen Turkish BERT document representation to the decision-history baseline change next-meeting direction macro-F1, action-size MAE or probability quality on the same 37 chronological origins?

### Qwen feature ablations

Which part of the archived Qwen representation contributes to forecasting behaviour?

- inflation-pressure score;
- demand-pressure score;
- forward-policy-bias score;
- all three scores without missingness indicators;
- missingness indicators without scores; or
- the original three scores plus three indicators.

## Fixed evaluation design

- Target: explicitly announced one-week repo action at the next recorded meeting.
- Classes: cut, hold and hike.
- Forecast origin: end of the current decision publication day.
- Evaluation origins: all eligible origins from 1 January 2023 onward (37 origins).
- Training set: earlier origins whose next-meeting target had been published by the current forecast origin.
- Direction model: L2-regularised multinomial logistic regression.
- Magnitude model: ridge regression.
- Regularisation coefficient: 0.1 for every arm.
- Primary metrics: macro-F1 for direction and MAE in basis points for magnitude.
- Additional metric: multiclass Brier score.
- Uncertainty: 10,000 paired circular-block resamples, block length four, with the same sampled indices applied to every arm.

## Frozen contextual representation

The configured transformer is not fine-tuned. Decision texts are tokenised without truncating the document to a single model window. The token sequence is divided into overlapping chunks. Each chunk is mean-pooled over non-padding token states, document vectors are obtained by averaging chunk vectors, and the document vector is L2-normalised.

The embedding model is fixed across all folds, but its numeric columns are standardised using only each fold’s training rows. The model’s resolved repository commit is recorded after loading. Because the encoder may have encountered historical policy text during pretraining, this experiment is a contextual-feature benchmark rather than a fully point-in-time-pretrained language-model experiment.

## Qwen ablations

Archived invalid Qwen outputs remain missing. For score-bearing arms, medians are learned only from the current training fold. Single-axis arms include that axis’s missingness indicator. The scores-only arm omits all indicators. The missing-only arm contains no semantic scores.

The ablations were defined after the original results were available. They are robustness analyses and cannot be presented as preregistered primary comparisons.

## Interpretation rules

- A higher point estimate is not sufficient for a superiority claim.
- Direction and magnitude findings are reported separately.
- If the missing-only arm performs similarly to a score arm, output validity may be carrying predictive structure and this must be discussed.
- The transformer result must be reported with its exact resolved model commit and pooling procedure.
- Results remain conditional on the 37 observed origins and the Turkish institutional setting.