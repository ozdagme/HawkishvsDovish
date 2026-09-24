#!/usr/bin/env python3
"""Export de-identified source, expert-label, and Qwen-feature audit tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

AXES = ("inflation_pressure", "demand_pressure", "forward_policy_bias")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(archive: zipfile.ZipFile, member: str) -> list[dict]:
    return [
        json.loads(line)
        for line in archive.read(member).decode("utf-8").splitlines()
        if line.strip()
    ]


def opaque_item_id(origin_id: str) -> str:
    digest = hashlib.sha256(
        ("human-review-order-20260913|" + origin_id).encode("utf-8")
    ).hexdigest()
    return "T-" + digest[:12]


def split_name(index: int) -> str:
    return "train" if index < 80 else "validation" if index < 98 else "test"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.prepared) as archive:
        features = read_jsonl(archive, "features_decision_origin.jsonl")[:-1]
    with zipfile.ZipFile(args.calendar) as archive:
        calendar = {
            row["meeting_date"]: row
            for row in json.loads(archive.read("official_calendar.json"))
        }
    labels = json.loads(args.labels.read_text(encoding="utf-8-sig"))["annotations"]

    source_rows = []
    label_rows = []
    for index, feature in enumerate(features):
        item_id = opaque_item_id(feature["origin_id"])
        source_rows.append(
            {
                "index": index,
                "origin_id": feature["origin_id"],
                "origin_date": feature["origin_date"],
                "split": split_name(index),
                "decision_document_id": feature["decision_document_id"],
                "decision_url": calendar[feature["origin_date"]]["decision_url"],
                "decision_text_sha256": feature["decision_text_sha256"],
            }
        )
        annotation = labels[item_id]
        row = {
            "index": index,
            "origin_id": feature["origin_id"],
            "origin_date": feature["origin_date"],
            "split": split_name(index),
            "item_id": item_id,
        }
        for axis in AXES:
            value = annotation["axes"][axis]
            row[f"{axis}_score"] = value["score"]
            row[f"{axis}_evidence"] = value["evidence"]
            row[f"{axis}_zero_reason"] = value["zero_reason"]
            row[f"{axis}_note"] = value["note"]
        label_rows.append(row)

    qwen_rows = []
    by_origin = {feature["origin_id"]: feature for feature in features}
    with zipfile.ZipFile(args.qwen) as archive:
        for index, feature in enumerate(features):
            record = json.loads(archive.read(f"records/{feature['origin_id']}.json"))
            parsed = None
            try:
                parsed = json.loads(record["raw_response"].strip())
            except (json.JSONDecodeError, TypeError):
                parsed = None
            row = {
                "index": index,
                "origin_id": feature["origin_id"],
                "origin_date": feature["origin_date"],
                "split": split_name(index),
                "status": record["status"],
                "ended_with_eos": record["ended_with_eos"],
                "prompt_sha256": record["prompt_sha256"],
            }
            for axis in AXES:
                item = parsed.get(axis, {}) if isinstance(parsed, dict) else {}
                row[f"{axis}_score"] = item.get("score", "")
                row[f"{axis}_evidence"] = item.get("evidence", "")
            qwen_rows.append(row)

    if set(by_origin) != {row["origin_id"] for row in source_rows}:
        raise ValueError("Source export lost an origin")
    if len(label_rows) != 122 or len(qwen_rows) != 122:
        raise ValueError("Expected 122 exported rows")
    write_csv(args.output_dir / "tcmb_statement_source_index.csv", source_rows)
    write_csv(args.output_dir / "finance_expert_labels.csv", label_rows)
    write_csv(args.output_dir / "qwen_features_public.csv", qwen_rows)
    metadata = {
        "rows": 122,
        "personal_information_included": False,
        "future_policy_outcomes_included": False,
        "expert_profile_included": False,
        "label_file_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "prepared_archive_sha256": hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        "calendar_archive_sha256": hashlib.sha256(args.calendar.read_bytes()).hexdigest(),
        "qwen_archive_sha256": hashlib.sha256(args.qwen.read_bytes()).hexdigest(),
    }
    (args.output_dir / "public_audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", **metadata}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())