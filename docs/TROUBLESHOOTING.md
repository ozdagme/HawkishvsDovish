# Troubleshooting

## `SHA-256 mismatch`

One of the four source archives was changed, renamed internally or recompressed from different content. Restore the exact files distributed with the repository. Do not update the expected hash simply to bypass the check.

## `Transformer experiment requires torch and transformers`

Install a PyTorch build compatible with your platform, then install `requirements.txt`.

## Hugging Face download failure

Check network access and the Hugging Face cache directory. If the model was downloaded elsewhere, set:

```bash
export HF_HOME=/path/to/huggingface/cache
```

Then rerun `python run.py embed`.

## CUDA out of memory

Lower `embedding.batch_size` in `configs/publication.json`, or set `embedding.device` to `cpu`. This changes runtime but not the mathematical document representation.

## Existing output directory

The pipeline does not silently overwrite results. Archive the existing directory or rerun intentionally with `--force`.

## `resolved_commit` is empty

Do not finalise manuscript reproducibility text. Record the installed `transformers` version, save the Hugging Face snapshot identifier manually and pin it in `embedding.revision` before the final run.

## Smoke results differ across machines

The history, TF-IDF, LSA and Qwen arms should agree to numerical tolerance. Minor floating-point differences can occur in optimisation. Class predictions and source counts must remain unchanged.
# Transformers 5: `prepare_for_model` attribute error

The current pipeline does not call the removed `prepare_for_model`,
`build_inputs_with_special_tokens`, or tokenizer-padding helpers. For the
frozen BERT encoder it constructs CLS/SEP/PAD sequences directly from stable
token-ID attributes. If one of these attribute errors appears, replace the local
`src/tcmb_repro/pipeline.py` with the current repository version and rerun
`python run.py embed`.