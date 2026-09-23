# TCMB financial-NLP extension results

**Status:** Post-hoc extension on previously observed dates; not an independent confirmatory test.

Embedding backend: `transformer`

| Arm | Correct/N | Macro-F1 | MAE (bp) | Brier |
|---|---:|---:|---:|---:|
| history | 23/37 | 0.6035 | 143.05 | 0.5234 |
| history_tfidf | 24/37 | 0.6321 | 139.84 | 0.5154 |
| history_lsa | 23/37 | 0.5961 | 140.91 | 0.5179 |
| history_transformer | 24/37 | 0.6345 | 158.01 | 0.6271 |
| history_qwen_all | 23/37 | 0.5772 | 138.31 | 0.5510 |
| history_qwen_inflation | 22/37 | 0.5560 | 142.46 | 0.5396 |
| history_qwen_demand | 23/37 | 0.5961 | 135.75 | 0.5278 |
| history_qwen_forward | 24/37 | 0.6269 | 145.07 | 0.5405 |
| history_qwen_scores_only | 24/37 | 0.6191 | 138.50 | 0.5371 |
| history_qwen_missing_only | 23/37 | 0.5961 | 142.15 | 0.5338 |

## Paired differences relative to history

| Candidate | Macro-F1 difference [95% interval] | MAE difference [95% interval] |
|---|---:|---:|
| history_tfidf | 0.0286 [0.0000, 0.0697] | -3.21 [-6.08, -0.61] |
| history_lsa | -0.0074 [-0.0837, 0.0571] | -2.14 [-4.16, -0.16] |
| history_transformer | 0.0310 [-0.0765, 0.1564] | 14.95 [-18.74, 48.53] |
| history_qwen_all | -0.0263 [-0.1248, 0.0778] | -4.74 [-18.71, 8.41] |
| history_qwen_inflation | -0.0475 [-0.1352, 0.0505] | -0.59 [-7.63, 6.28] |
| history_qwen_demand | -0.0074 [-0.0837, 0.0571] | -7.30 [-20.59, 5.35] |
| history_qwen_forward | 0.0234 [0.0000, 0.0833] | 2.01 [-2.68, 6.85] |
| history_qwen_scores_only | 0.0156 [-0.0778, 0.1102] | -4.55 [-18.35, 7.96] |
| history_qwen_missing_only | -0.0074 [-0.0837, 0.0571] | -0.90 [-4.02, 2.01] |

Hash embeddings, if present in a smoke-test directory, are software checks only and must not be reported.