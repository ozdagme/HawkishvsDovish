#!/usr/bin/env python3
"""Export a completed semantic run without model weights or new inference."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import zipfile


def validate_run(run_dir):
    split_path = run_dir / "split_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))["documents"]
    if len({row["id"] for row in split}) != len(split):
        raise ValueError("Duplicate document identities in split manifest")
    held_out = {row["id"]: row["split"] for row in split if row["split"] != "train"}
    if (len(held_out) != 42 or list(held_out.values()).count("test") != 24
            or list(held_out.values()).count("validation") != 18):
        raise ValueError("Expected 18 validation and 24 test documents")
    paths = sorted((run_dir / "predictions").rglob("*.json"))
    if not paths:
        state = "missing" if not (run_dir / "predictions").is_dir() else "empty"
        raise ValueError(
            f"{run_dir}: predictions directory is {state}; found 0 of 672 required predictions. "
            "Find the saved completed run with --discover --search-root outputs. "
            "This exporter cannot reconstruct missing responses; do not rerun training yet.")
    expected = {(family, seed, rag, identity) for family in ("qwen", "mistral")
                for seed in (None, 20260916, 20260917, 20260918)
                for rag in ("no_rag", "rag") for identity in held_out}
    identities, run_identities = set(), set()
    for path in paths:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        record = envelope["data"]
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != envelope["sha"]:
            raise ValueError(f"Corrupted stored prediction: {path}")
        if envelope["task"] != path.relative_to(run_dir).as_posix():
            raise ValueError(f"Prediction path does not match task: {path}")
        run_identities.add(envelope["identity"])
        identity = (record["family"], record["seed"], record["rag"], record["id"])
        if identity in identities or record["split"] != held_out.get(record["id"]):
            raise ValueError(f"Duplicate or mismatched prediction: {path}")
        identities.add(identity)
    if identities != expected or len(run_identities) != 1:
        raise ValueError(f"Incomplete or mixed semantic run: found {len(identities)} of 672 required predictions")
    required = ["split_manifest.json", "report.json", "results.csv", "axis_results.csv",
                "generation_summary.csv", "tasks/cpu.json", "tasks/bert.json"]
    paths.extend(run_dir / name for name in required)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths, identities


def discover_runs(search_root):
    if not search_root.is_dir():
        raise ValueError(f"Search directory does not exist: {search_root}; try --search-root .")
    candidates = sorted({path.parent for path in search_root.rglob("split_manifest.json")})
    ready = []
    for run_dir in candidates:
        count = sum(1 for _ in (run_dir / "predictions").rglob("*.json"))
        try:
            validate_run(run_dir)
        except (ValueError, OSError, KeyError, TypeError) as error:
            print(f"NOT_READY | {run_dir} | JSON files: {count}/672 | {error}")
        else:
            ready.append(run_dir)
            print(f"READY | {run_dir} | validated responses: 672/672")
            print(f"  --run-dir {shlex.quote(str(run_dir))}")
    if not candidates:
        print(f"No split_manifest.json found under {search_root}. Try --search-root .")
    print(f"Summary: {len(ready)} ready run(s), {len(candidates)} candidate(s). No model executed.")
    if not ready:
        print("Locate the original saved predictions or completed-run archive on your computer/backups. "
              "Do not replace them with newly generated responses.")
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=Path)
    mode.add_argument("--discover", action="store_true", help="List and validate saved runs without exporting")
    mode.add_argument("--auto", action="store_true", help="Export only if exactly one valid run is found")
    parser.add_argument("--search-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("SEMANTIK_KAYITLI_TAHMINLER.zip"))
    args = parser.parse_args()
    if any(path.is_absolute() for path in (args.run_dir, args.search_root, args.output) if path is not None):
        parser.error("Use paths relative to the project root")
    if args.discover or args.auto:
        ready = discover_runs(args.search_root)
        if args.discover:
            return
        if len(ready) != 1:
            raise ValueError(f"Automatic export requires exactly one validated run; found {len(ready)}. "
                             "If multiple runs are ready, choose the original publication run with --run-dir.")
        args.run_dir = ready[0]
    paths, identities = validate_run(args.run_dir)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(args.run_dir))
        archive.writestr("export_manifest.json", json.dumps({
            "status": "existing predictions only; no model execution",
            "llm_records": len(identities), "model_weights_included": False,
            "files_sha256": {path.relative_to(args.run_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in paths}}, indent=2))
    print(f"Exported 672 existing responses to {args.output}; no GPU or new training used.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        sys.exit(f"Export stopped: {error}")