#!/usr/bin/env python3
"""Audit frozen semantic responses and rescore both expert references without inference."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


AXES = ("inflation_pressure", "demand_pressure", "forward_policy_bias")
SEEDS = (20260916, 20260917, 20260918)
MISSING = 9
CONTRASTS = {
    "qwen_ft_effect": ("qwen_ft_no_rag", "qwen_base_no_rag"),
    "mistral_ft_effect": ("mistral_ft_no_rag", "mistral_base_no_rag"),
    "qwen_ft_rag_effect": ("qwen_ft_rag", "qwen_ft_no_rag"),
    "mistral_ft_rag_effect": ("mistral_ft_rag", "mistral_ft_no_rag"),
}
PUBLISHED_F1 = {
    "majority": 0.1728, "tfidf": 0.1728, "frozen_turkish_bert": 0.1728,
    "qwen_base_no_rag": 0.2766, "qwen_base_rag": 0.3057,
    "qwen_ft_no_rag": 0.4031, "qwen_ft_rag": 0.2896,
    "mistral_base_no_rag": 0.3886, "mistral_base_rag": 0.2340,
    "mistral_ft_no_rag": 0.4595, "mistral_ft_rag": 0.4030,
}
PUBLISHED_CONTRASTS = {
    "qwen_ft_effect": (0.1265, 0.0005, 0.2063),
    "mistral_ft_effect": (0.0709, -0.0628, 0.1673),
    "qwen_ft_rag_effect": (-0.1135, -0.2019, 0.0201),
    "mistral_ft_rag_effect": (-0.0565, -0.1401, 0.0178),
}


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON keys")
    return result


def validate_prediction(prediction):
    if not isinstance(prediction, dict) or set(prediction) != set(AXES):
        raise ValueError("Expected exactly the three semantic axes")
    if any(type(value) is not int or value not in (-1, 0, 1) for value in prediction.values()):
        raise ValueError("Semantic values must be integers -1, 0 or 1; no coercion")
    return prediction


def first_object(raw):
    start = raw.find("{")
    if start < 0:
        raise ValueError("No initial JSON object in stored response")
    prediction, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(raw[start:])
    validate_prediction(prediction)
    strict = not raw[:start].strip() and not raw[start + end:].strip()
    return prediction, strict


def decode_envelope(content, name):
    envelope = json.loads(content)
    encoded = json.dumps(envelope["data"], ensure_ascii=False, sort_keys=True,
                         allow_nan=False, indent=2).encode("utf-8")
    if envelope["task"] != name or sha256(encoded) != envelope["sha"]:
        raise ValueError(f"Stored envelope integrity mismatch: {name}")
    return envelope


def load_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate ZIP members")
        export = json.loads(archive.read("export_manifest.json"))
        if set(names) != set(export["files_sha256"]) | {"export_manifest.json"}:
            raise ValueError("ZIP members differ from export manifest")
        for name, expected in export["files_sha256"].items():
            if sha256(archive.read(name)) != expected:
                raise ValueError(f"Export hash mismatch: {name}")
        documents = json.loads(archive.read("split_manifest.json"))["documents"]
        if len(documents) != 122 or len({row["id"] for row in documents}) != 122:
            raise ValueError("Expected 122 unique split documents")
        if [row["split"] for row in documents] != ["train"] * 80 + ["validation"] * 18 + ["test"] * 24:
            raise ValueError("Split is not chronological 80/18/24")
        if [row["date"] for row in documents] != sorted(row["date"] for row in documents):
            raise ValueError("Document dates are not chronological")
        held_out = documents[80:]
        expected = {(family, seed, rag, row["id"]) for family in ("qwen", "mistral")
                    for seed in (None, *SEEDS) for rag in ("no_rag", "rag") for row in held_out}
        records, identities, target_hashes, parsed_rows = {}, set(), {}, []
        for name in names:
            if not name.startswith("predictions/"):
                continue
            envelope = decode_envelope(archive.read(name), name)
            identities.add(envelope["identity"])
            record = envelope["data"]
            key = (record["family"], record["seed"], record["rag"], record["id"])
            if key not in expected or key in records:
                raise ValueError(f"Unexpected or duplicate prediction identity: {name}")
            document = next(row for row in held_out if row["id"] == record["id"])
            if record["split"] != document["split"]:
                raise ValueError(f"Mismatched split: {name}")
            prediction, strict = first_object(record["raw"])
            if type(record["schema_valid"]) is not bool or strict != record["schema_valid"]:
                raise ValueError(f"Strict-format status mismatch: {name}")
            if strict and prediction != record["prediction"]:
                raise ValueError(f"Strict prediction differs from raw object: {name}")
            target_key = (record["family"], record["id"])
            target_sha = record["audit"]["target_sha"]
            if target_hashes.setdefault(target_key, target_sha) != target_sha:
                raise ValueError(f"Target differs across seeds/conditions: {name}")
            records[key] = {**record, "parsed": prediction}
            parsed_rows.append({"group": f"{record['family']}_{'base' if record['seed'] is None else 'ft'}_{record['rag']}",
                                "seed": record["seed"], "origin_id": record["id"],
                                "origin_date": document["date"], "split": record["split"],
                                "strict_json": strict, **prediction, "source_sha256": sha256(archive.read(name))})
        if set(records) != expected or export["llm_records"] != 672 or len(identities) != 1:
            raise ValueError("Incomplete or mixed 672-record run")
        groups, strict_groups = {}, {}
        for name, keys in (("tasks/cpu.json", {"majority": "majority", "tfidf": "tfidf"}),
                           ("tasks/bert.json", {"frozen_turkish_bert": "predictions"})):
            envelope = decode_envelope(archive.read(name), name)
            if envelope["identity"] not in identities:
                raise ValueError(f"Baseline belongs to a different run: {name}")
            for group, key in keys.items():
                predictions = envelope["data"][key]
                if len(predictions) != 42:
                    raise ValueError("Baseline must contain 42 held-out predictions")
                for prediction in predictions:
                    validate_prediction(prediction)
                groups[group] = [[prediction for prediction in predictions[18:]]]
                strict_groups[group] = groups[group]
        for family in ("qwen", "mistral"):
            for mode, seeds in (("base", (None,)), ("ft", SEEDS)):
                for rag in ("no_rag", "rag"):
                    group = f"{family}_{mode}_{rag}"
                    groups[group] = [[records[(family, seed, rag, row["id"])]["parsed"]
                                      for row in documents[98:]] for seed in seeds]
                    strict_groups[group] = [[records[(family, seed, rag, row["id"])]["prediction"]
                                             for row in documents[98:]] for seed in seeds]
        stored_results = list(csv.DictReader(io.StringIO(archive.read("results.csv").decode("utf-8-sig"))))
        return documents, groups, strict_groups, parsed_rows, stored_results, identities.pop()


def array_predictions(runs):
    return np.asarray([[[prediction[axis] if prediction[axis] is not None else MISSING
                         for axis in AXES] for prediction in run] for run in runs])


def score(runs, truth):
    valid = truth != MISSING
    if np.any(valid.sum(axis=0) == 0):
        raise ValueError("No reference labels on an axis")
    accuracy = ((runs == truth) & valid).sum(axis=1) / valid.sum(axis=0)
    f1 = np.zeros((len(runs), len(AXES)))
    for category in (-1, 0, 1):
        denominator = (truth == category).sum(axis=0) + ((runs == category) & valid).sum(axis=1)
        numerator = 2 * ((runs == category) & (truth == category)).sum(axis=1)
        f1 += np.divide(numerator, denominator, out=np.zeros_like(f1), where=denominator > 0) / 3
    return {"macro_f1": float(f1.mean()), "axis_mean_accuracy": float(accuracy.mean()),
            "seed_sd": float(np.std(f1.mean(axis=1), ddof=1)) if len(runs) > 1 else None,
            "axis_f1": f1.mean(axis=0).tolist(), "axis_accuracy": accuracy.mean(axis=0).tolist(),
            "axis_n": valid.sum(axis=0).tolist()}


def bootstrap_scores(runs, truth, draws):
    gold, predictions = truth[draws][None, ...], runs[:, draws, :]
    values = np.zeros((len(runs), len(draws), len(AXES)))
    for category in (-1, 0, 1):
        denominator = (gold == category).sum(axis=2) + ((predictions == category) & (gold != MISSING)).sum(axis=2)
        numerator = 2 * ((predictions == category) & (gold == category)).sum(axis=2)
        values += np.divide(numerator, denominator, out=np.zeros_like(values), where=denominator > 0) / 3
    return values.mean(axis=(0, 2))


def read_labels(path, documents):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {row["origin_id"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {row["id"] for row in documents}:
        raise ValueError(f"Reference document identities mismatch: {path}")
    result = []
    for document in documents:
        row = indexed[document["id"]]
        if row["origin_date"] != document["date"] or row.get("split", document["split"]) != document["split"]:
            raise ValueError(f"Reference date/split mismatch: {path}")
        labels = [row[axis + "_score"] for axis in AXES]
        if any(value not in ("-1", "0", "1", "U") for value in labels):
            raise ValueError(f"Unknown reference class: {path}")
        result.append([MISSING if value == "U" else int(value) for value in labels])
    return np.asarray(result)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    support = Path(__file__).parent
    parser.add_argument("--archive", type=Path, default=Path("prism-uploads/SEMANTIK_KAYITLI_TAHMINLER.zip"))
    parser.add_argument("--original", type=Path, default=support / "finance_expert_labels.csv")
    parser.add_argument("--second", type=Path, default=support / "second_expert_audit/second_expert_labels_test.csv")
    parser.add_argument("--output", type=Path, default=support / "second_expert_rescoring")
    args = parser.parse_args()
    documents, groups, strict_groups, parsed_rows, stored_results, run_identity = load_archive(args.archive)
    original = read_labels(args.original, documents)[98:]
    second = read_labels(args.second, documents[98:])
    common = (original != MISSING) & (second != MISSING)
    references = {"original_full": original, "second_full": second,
                  "original_common": np.where(common, original, MISSING),
                  "second_common": np.where(common, second, MISSING)}
    stored_test = {row["group"]: row for row in stored_results if row["split"] == "test"}
    arrays = {group: array_predictions(runs) for group, runs in groups.items()}
    for group, runs in strict_groups.items():
        result = score(array_predictions(runs), original)
        for field in ("macro_f1", "axis_mean_accuracy"):
            if not np.isclose(result[field], float(stored_test[group][field]), atol=1e-12, rtol=0):
                raise ValueError(f"Original strict report mismatch: {group}/{field}")
    starts = np.random.default_rng(SEEDS[0]).integers(0, 24, size=(2000, 8))
    draws = ((starts[:, :, None] + np.arange(3)) % 24).reshape(2000, 24)
    metrics, axes, contrasts = [], [], []
    for reference, truth in references.items():
        points, samples = {}, {}
        for group, runs in arrays.items():
            result = score(runs, truth)
            points[group] = result["macro_f1"]
            if reference == "original_full" and round(points[group], 4) != PUBLISHED_F1[group]:
                raise ValueError(f"Published first-object F1 mismatch: {group}")
            metrics.append({"reference": reference, "group": group, "n_documents": 24, "n_runs": len(runs),
                            **{key: result[key] for key in ("macro_f1", "axis_mean_accuracy", "seed_sd")}})
            for index, axis in enumerate(AXES):
                axes.append({"reference": reference, "group": group, "axis": axis,
                             "n": result["axis_n"][index], "macro_f1": result["axis_f1"][index],
                             "accuracy": result["axis_accuracy"][index]})
            samples[group] = bootstrap_scores(runs, truth, draws)
        for name, (candidate, baseline) in CONTRASTS.items():
            estimate = points[candidate] - points[baseline]
            low, high = np.quantile(samples[candidate] - samples[baseline], [.025, .975])
            if reference == "original_full" and tuple(round(float(value), 4) for value in (estimate, low, high)) != PUBLISHED_CONTRASTS[name]:
                raise ValueError(f"Published paired interval mismatch: {name}")
            contrasts.append({"reference": reference, "contrast": name, "estimate": estimate,
                              "interval95_low": float(low), "interval95_high": float(high),
                              "status": "exploratory reference sensitivity; stored-response conditional"})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("model_metrics.csv", metrics), ("axis_metrics.csv", axes),
                       ("paired_contrasts.csv", contrasts), ("parsed_responses.csv", parsed_rows)):
        write_csv(args.output / name, rows)
    np.savetxt(args.output / "bootstrap_indices.csv", draws, delimiter=",", fmt="%d")
    outputs = sorted(args.output.glob("*.csv"))
    manifest = {"date": "2026-09-30", "status": "frozen responses rescored; no inference, refitting or adjudication",
                "run_identity": run_identity, "llm_records": 672, "first_objects_valid": 672,
                "strict_json": {"base_valid": sum(row["strict_json"] for row in parsed_rows if row["seed"] is None),
                                "base_n": sum(row["seed"] is None for row in parsed_rows),
                                "fine_tuned_valid": sum(row["strict_json"] for row in parsed_rows if row["seed"] is not None),
                                "fine_tuned_n": sum(row["seed"] is not None for row in parsed_rows)},
                "reproduction": "Original strict report, 11 published first-object F1 values and four paired intervals match",
                "parser": "Decode first object starting at first opening brace; reject duplicate keys, noninteger values or altered axes; no repair",
                "reference_axis_n": {name: (truth != MISSING).sum(axis=0).tolist() for name, truth in references.items()},
                "class_order": [-1, 0, 1], "axis_order": list(AXES),
                "bootstrap": {"seed": SEEDS[0], "draws": 2000, "circular_block_length": 3,
                              "unit": "document; same indices for all references/models; average seed metrics, not labels"},
                "limitations": "Same reused 24 documents; new annotations, not new outcomes. Conditional exploratory intervals, not independent confirmation or reference adjudication.",
                "script_sha256": sha256(Path(__file__).read_bytes()),
                "input_sha256": {str(path): sha256(path.read_bytes()) for path in (args.archive, args.original, args.second)},
                "output_sha256": {path.name: sha256(path.read_bytes()) for path in outputs}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest["reproduction"])
    for row in metrics:
        if row["reference"] == "second_full":
            print(row["group"], f"F1={row['macro_f1']:.4f}", f"accuracy={row['axis_mean_accuracy']:.4f}")
    for row in contrasts:
        if row["reference"].startswith("second"):
            print(row["reference"], row["contrast"], f"{row['estimate']:.4f} [{row['interval95_low']:.4f}, {row['interval95_high']:.4f}]")


if __name__ == "__main__":
    main()