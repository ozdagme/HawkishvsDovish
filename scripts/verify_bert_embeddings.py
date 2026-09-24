#!/usr/bin/env python3
"""Validate the regenerated BERT document-embedding cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcmb_repro.pipeline import load_dataset, load_embeddings, read_json  # noqa: E402


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--config", default="configs/publication.json")
    parser.add_argument("--embeddings", default="data/derived/bert_embeddings.npz")
    args = parser.parse_args()

    config = read_json(resolve(args.config))
    dataset = load_dataset(resolve(args.raw_dir), config["evaluation_origin_start"])
    matrix, metadata = load_embeddings(resolve(args.embeddings), dataset)
    embedding_config = config["embedding"]
    expected_commit = embedding_config.get("revision")
    expected_hash = embedding_config.get("expected_matrix_sha256")
    if not expected_hash or not expected_commit or expected_commit == "main":
        raise SystemExit(
            "The BERT verification configuration is outdated. Replace "
            "configs/publication.json with the version supplied in "
            "BERT_NESTED_REPRODUCTION_CODE.zip, regenerate the embeddings "
            "with `python run.py embed --force`, and rerun this validator."
        )
    report = {
        "status": "exact_match"
        if metadata.get("resolved_commit") == expected_commit
        and metadata.get("matrix_sha256") == expected_hash
        else "review_required",
        "shape": list(matrix.shape),
        "backend": metadata.get("backend"),
        "resolved_commit": metadata.get("resolved_commit"),
        "expected_commit": expected_commit,
        "matrix_sha256": metadata.get("matrix_sha256"),
        "expected_matrix_sha256": expected_hash,
        "commit_matches": metadata.get("resolved_commit") == expected_commit,
        "matrix_matches": metadata.get("matrix_sha256") == expected_hash,
        "metadata": metadata,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "exact_match" else 2


if __name__ == "__main__":
    raise SystemExit(main())