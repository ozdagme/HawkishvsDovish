# Data files

The `raw` directory contains four immutable analysis archives identified by SHA-256 in `src/tcmb_repro/pipeline.py`.

| File | Purpose |
|---|---|
| `data.zip` | Processed TCMB documents and associated numeric records |
| `tcmb_next_meeting_prepare_v1.zip` | Decision-origin texts, next-meeting targets and timing audit |
| `tcmb_calendar_audit_v1_repeat2.zip` | Preserved official-calendar audit |
| `tcmb_qwen_features_v1.zip` | Archived Qwen requests, raw outputs and validation records |

The analysis uses decision statements as model text. Post-meeting summaries are retained only for source and timing checks and are not supplied to the forecasting models or retrieval conditions.

Third-party market-source files are not redistributed by this new experimental pipeline because the contextual and Qwen-ablation experiments do not require them. The `data.zip` archive is retained to verify the frozen research cohort and provenance.

The code license does not override the terms of the underlying public or third-party data sources. Users are responsible for respecting source-specific conditions when redistributing raw data.