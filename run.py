#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tcmb_repro.pipeline import (  # noqa: E402
    build_embeddings,
    check_repository,
    read_json,
    run_experiment,
    stage_inputs,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="TCMB financial-NLP reproducibility pipeline")
    command.add_argument(
        "action",
        choices=("stage-data", "check", "embed", "run", "all", "smoke"),
        help="Pipeline stage",
    )
    command.add_argument("--config", default="configs/publication.json")
    command.add_argument("--raw-dir", default="data/raw")
    command.add_argument("--embeddings", default="data/derived/bert_embeddings.npz")
    command.add_argument("--output", default="outputs/publication_run")
    command.add_argument(
        "--source-dir",
        default="..",
        help="Folder searched recursively by stage-data for the four frozen input archives",
    )
    command.add_argument("--backend", choices=("transformer", "hash"))
    command.add_argument("--force", action="store_true", help="Replace the selected output directory/cache")
    return command


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parser().parse_args()
    config = read_json(resolve(args.config))
    raw_dir = resolve(args.raw_dir)
    embeddings = resolve(args.embeddings)
    output = resolve(args.output)

    if args.action == "stage-data":
        report = stage_inputs(resolve(args.source_dir), raw_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"DATA_READY: {raw_dir}")
        return 0

    if args.action == "check":
        report = check_repository(raw_dir, config)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.action == "smoke":
        embeddings = ROOT / "data/derived/smoke_hash_embeddings.npz"
        output = ROOT / "outputs/smoke"
        if args.force:
            embeddings.unlink(missing_ok=True)
            shutil.rmtree(output, ignore_errors=True)
        metadata = build_embeddings(raw_dir, embeddings, config, backend="hash")
        run = run_experiment(
            raw_dir,
            output,
            embeddings,
            config,
            allow_nonpublication_backend=True,
        )
        write_json(output / "SMOKE_ONLY.json", {"embedding": metadata, "run": run})
        print(f"SMOKE_OK: {output}")
        return 0

    if args.action in ("embed", "all"):
        if embeddings.exists() and not args.force:
            raise FileExistsError(f"Embedding cache exists: {embeddings}; use --force to replace it")
        embeddings.unlink(missing_ok=True)
        metadata = build_embeddings(raw_dir, embeddings, config, backend=args.backend)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        if args.action == "embed":
            return 0

    if args.action in ("run", "all"):
        if output.exists() and not args.force:
            raise FileExistsError(f"Output directory exists: {output}; use --force to replace it")
        if args.force:
            shutil.rmtree(output, ignore_errors=True)
        metadata = run_experiment(raw_dir, output, embeddings, config)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        print(f"COMPLETE: {output}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())