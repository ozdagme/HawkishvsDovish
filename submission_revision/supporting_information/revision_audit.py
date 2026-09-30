#!/usr/bin/env python3
"""Stored-forecast diagnostics and blinded second-expert assessment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import zipfile

import numpy as np


CLASSES = ("cut", "hold", "hike")
AXES = ("inflation_pressure", "demand_pressure", "forward_policy_bias")
PERIODS = (
    ("before_2023_06_22", "0001-01-01", "2023-06-21"),
    ("2023_06_22_to_2024_03_21", "2023-06-22", "2024-03-21"),
    ("2024_03_22_to_2024_12_25", "2024-03-22", "2024-12-25"),
    ("2024_12_26_onward", "2024-12-26", "9999-12-31"),
)
RUBRIC = """# Bağımsız uzman etiketleme kuralları

Her metni yalnız kendi içeriğine göre değerlendirin. İlk uzman etiketlerine,
model çıktılarına veya çalışmanın sonuçlarına bakmayın. Sonraki faiz kararını
tahmin etmeyin. Üç ekseni birbirinden bağımsız değerlendirin.

| Eksen | -1 | 0 | +1 |
|---|---|---|---|
| Enflasyon baskısı | Azalıyor veya zayıflıyor | Karışık, dengeli ya da belirtilmemiş | Artıyor veya güçleniyor |
| Talep baskısı | Zayıf veya zayıflıyor | Karışık, dengeli ya da belirtilmemiş | Güçlü veya güçleniyor |
| İleri politika eğilimi | Açık geleceğe yönelik gevşeme/indirim | Yönsüz, belirsiz veya yalnız mevcut sıkılığın korunması | Açık geleceğe yönelik ek sıkılaşma/artırım |

Mevcut toplantının faiz kararı tek başına ileri politika etiketi değildir.
Mevcut sıkılığın korunması yeni bir sıkılaşma sinyali sayılmaz. Metinde eksenin
hiç belirtilmemesi 0'dır; gerçekten kararsızsanız U seçin ve gerekçe yazın.
U analizde ayrı ekonomik sınıf veya 0 olarak kullanılmaz. Mümkünse her karar
için kısa bir metinsel kanıt kaydedin. Rubrik sorularını başlamadan sorun;
etiketleme sırasında yeni sınıf tanımları geliştirmeyin.
"""


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_csv_identity(path: Path, expected_sha256: str) -> dict:
    actual_sha256 = file_hash(path)
    if actual_sha256 == expected_sha256:
        return {"status": "byte_exact", "actual_sha256": actual_sha256,
                "expected_sha256": expected_sha256}
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Empty frozen CSV: {path}")
    for newline in ("\r\n", "\n"):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)
        serialized = stream.getvalue()
        for trailing in (serialized, serialized.removesuffix(newline)):
            for prefix in ("\ufeff", ""):
                digest = hashlib.sha256((prefix + trailing).encode("utf-8")).hexdigest()
                if digest == expected_sha256:
                    return {"status": "csv_serialization_equivalent", "actual_sha256": actual_sha256,
                            "expected_sha256": expected_sha256, "matching_newline": repr(newline),
                            "matching_bom": bool(prefix), "matching_final_newline": trailing == serialized}
    raise ValueError(f"Frozen CSV content or identity changed: {path}")


def unique_rows(rows: list[dict], key: str) -> dict:
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return indexed


def prepared_records(path: Path) -> tuple[dict, dict]:
    with zipfile.ZipFile(path) as archive:
        features = [json.loads(line) for line in archive.read(
            "features_decision_origin.jsonl").decode().splitlines() if line]
        targets = [json.loads(line) for line in archive.read(
            "evaluation_only_targets.jsonl").decode().splitlines() if line]
    feature_map = unique_rows(features, "origin_id")
    target_map = unique_rows(targets, "origin_id")
    if set(feature_map) != set(target_map):
        raise ValueError("Prepared feature and target identities differ")
    for feature in features:
        digest = hashlib.sha256(feature["decision_text"].encode()).hexdigest()
        if digest != feature["decision_text_sha256"]:
            raise ValueError(f"Statement hash mismatch: {feature['origin_id']}")
        publication = target_map[feature["origin_id"]]["target_publication_date"]
        if publication is None and target_map[feature["origin_id"]]["label_status"] != "unobserved":
            raise ValueError("Only an explicitly unobserved target may lack publication date")
        if publication is not None and publication <= feature["origin_date"]:
            raise ValueError("Target must be published after origin")
    return feature_map, target_map


def macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for category in CLASSES:
        denominator = int(np.sum(truth == category) + np.sum(predicted == category))
        numerator = 2 * int(np.sum((truth == category) & (predicted == category)))
        scores.append(numerator / denominator if denominator else 0.0)
    return float(np.mean(scores))


def metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate an empty cohort")
    truth = np.asarray([row["true_class"] for row in rows])
    predicted = np.asarray([row["predicted_class"] for row in rows])
    probabilities = np.asarray([json.loads(row["probabilities"]) for row in rows])
    if probabilities.shape != (len(rows), 3) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Invalid three-class probability matrix")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    if set(truth) - set(CLASSES) or set(predicted) - set(CLASSES):
        raise ValueError("Unknown direction class")
    observed = np.column_stack([truth == category for category in CLASSES])
    action_errors = [abs(float(row["true_action_bp"]) - float(row["predicted_action_bp"])) for row in rows]
    if not all(math.isfinite(error) for error in action_errors):
        raise ValueError("Nonfinite magnitude error")
    return {"n": len(rows), "cut_n": int(np.sum(truth == "cut")),
            "hold_n": int(np.sum(truth == "hold")), "hike_n": int(np.sum(truth == "hike")),
            "correct": int(np.sum(truth == predicted)), "macro_f1": macro_f1(truth, predicted),
            "mae_bp": float(np.mean(action_errors)),
            "brier": float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))}


def pair_rows(rows: list[dict], reference: str, candidate: str) -> tuple[list, list]:
    baseline = unique_rows([row for row in rows if row["arm"] == reference], "origin_id")
    augmented = unique_rows([row for row in rows if row["arm"] == candidate], "origin_id")
    if not baseline or set(baseline) != set(augmented):
        raise ValueError(f"Unmatched comparison: {reference}, {candidate}")
    identities = sorted(baseline, key=lambda identity: baseline[identity]["origin_date"])
    for identity in identities:
        for field in ("origin_date", "true_class", "true_action_bp"):
            if baseline[identity][field] != augmented[identity][field]:
                raise ValueError(f"Paired {field} differs at {identity}")
    return [baseline[identity] for identity in identities], [augmented[identity] for identity in identities]


def influence(rows: list[dict], reference: str, candidate: str, analysis: str) -> tuple[list, dict]:
    baseline, augmented = pair_rows(rows, reference, candidate)
    full = metrics(augmented)["macro_f1"] - metrics(baseline)["macro_f1"]
    details = []
    for index, row in enumerate(baseline):
        difference = (metrics(augmented[:index] + augmented[index + 1:])["macro_f1"] -
                      metrics(baseline[:index] + baseline[index + 1:])["macro_f1"])
        details.append({"analysis": analysis, "reference": reference, "candidate": candidate,
                        "omitted_origin_id": row["origin_id"], "omitted_origin_date": row["origin_date"],
                        "n_remaining": len(baseline) - 1, "macro_f1_difference": difference,
                        "change_from_full_difference": difference - full})
    differences = [row["macro_f1_difference"] for row in details]
    return details, {"analysis": analysis, "reference": reference, "candidate": candidate,
                     "n": len(baseline), "full_difference": full,
                     "leave_one_out_min": min(differences), "leave_one_out_max": max(differences),
                     "maximum_absolute_change": max(abs(value - full) for value in differences)}


def audit(args: argparse.Namespace) -> None:
    labels = read_csv(args.support / "finance_expert_labels.csv")
    unique_rows(labels, "origin_id")
    features, targets = prepared_records(args.prepared)
    nested = read_csv(args.support / "nested_tuning_predictions.csv")
    with zipfile.ZipFile(args.fixed) as archive:
        fixed = list(csv.DictReader(io.StringIO(archive.read("predictions.csv").decode("utf-8-sig"))))
        expected_fixed = {row["arm"]: row for row in csv.DictReader(io.StringIO(
            archive.read("metrics.csv").decode("utf-8-sig")))}
    expected_nested = {row["arm"]: row for row in read_csv(args.support / "nested_tuning_metrics.csv")}
    args.output.mkdir(parents=True, exist_ok=True)
    supports = []
    for split in ("train", "validation", "test"):
        cohort = [row for row in labels if row["split"] == split]
        for axis in AXES:
            values = [row[axis + "_score"] for row in cohort]
            if set(values) - {"-1", "0", "1", "U"}:
                raise ValueError(f"Invalid reference labels: {split}, {axis}")
            supports.append({"split": split, "axis": axis, "documents": len(cohort),
                             "minus_one": values.count("-1"), "zero": values.count("0"),
                             "plus_one": values.count("1"), "uncertain": values.count("U"),
                             "evaluated_n": sum(value != "U" for value in values)})
    write_csv(args.output / "class_support.csv", supports)
    recalculated = []
    for analysis, rows, expected in (("fixed", fixed, expected_fixed), ("nested", nested, expected_nested)):
        for row in rows:
            feature, target = features[row["origin_id"]], targets[row["origin_id"]]
            if (row["origin_date"] != feature["origin_date"] or row["true_class"] != target["direction"] or
                    float(row["true_action_bp"]) != float(target["action_bp"])):
                raise ValueError("Prediction does not match prepared source")
        for arm in sorted({row["arm"] for row in rows}):
            cohort = [row for row in rows if row["arm"] == arm]
            unique_rows(cohort, "origin_id")
            result = metrics(cohort)
            for computed, saved in (("macro_f1", "macro_f1"), ("mae_bp", "mae_bp"),
                                    ("brier", "brier_multiclass"), ("correct", "correct"), ("n", "n")):
                if not math.isclose(float(result[computed]), float(expected[arm][saved]), abs_tol=1e-9, rel_tol=1e-9):
                    raise ValueError(f"Stored metric mismatch: {analysis}, {arm}, {computed}")
            recalculated.append({"analysis": analysis, "arm": arm, **result})
    write_csv(args.output / "recomputed_metrics.csv", recalculated)
    period_rows = []
    for name, start, end in PERIODS:
        for arm in ("history", "proxy_market", "proxy_market_lsa"):
            cohort = [row for row in nested if row["arm"] == arm and
                      start <= targets[row["origin_id"]]["target_publication_date"] <= end]
            if cohort:
                dates = [targets[row["origin_id"]]["target_publication_date"] for row in cohort]
                period_rows.append({"period": name, "target_start": min(dates), "target_end": max(dates),
                                    "arm": arm, **metrics(cohort)})
    write_csv(args.output / "target_subperiod_metrics.csv", period_rows)
    details, summaries = [], []
    for analysis, rows, reference, candidate in (
            ("fixed", fixed, "history", "history_tfidf"),
            ("fixed", fixed, "history", "history_lsa"),
            ("fixed", fixed, "history", "history_transformer"),
            ("nested", nested, "proxy_market", "proxy_market_lsa")):
        detail, summary = influence(rows, reference, candidate, analysis)
        details.extend(detail)
        summaries.append(summary)
    write_csv(args.output / "leave_one_out_details.csv", details)
    write_csv(args.output / "leave_one_out_summary.csv", summaries)
    baseline, augmented = pair_rows(nested, "proxy_market", "proxy_market_lsa")
    changes = []
    for base, candidate in zip(baseline, augmented):
        if base["predicted_class"] != candidate["predicted_class"]:
            changes.append({"origin_id": base["origin_id"], "origin_date": base["origin_date"],
                            "target_date": targets[base["origin_id"]]["target_publication_date"],
                            "true_class": base["true_class"], "market_prediction": base["predicted_class"],
                            "market_lsa_prediction": candidate["predicted_class"],
                            "market_correct": base["true_class"] == base["predicted_class"],
                            "market_lsa_correct": candidate["true_class"] == candidate["predicted_class"]})
    write_csv(args.output / "market_lsa_changed_origins.csv", changes,
              ["origin_id", "origin_date", "target_date", "true_class", "market_prediction",
               "market_lsa_prediction", "market_correct", "market_lsa_correct"])
    manifest = {"analysis_status": "post-hoc descriptive audit; no refitting or new confirmation",
                "script_sha256": file_hash(Path(__file__)), "numpy_version": np.__version__,
                "period_assignment": "target publication date; retrospective calendar partitions, not model inputs",
                "period_definitions": PERIODS, "class_order": CLASSES,
                "leave_one_out": "omit one paired evaluation origin; keep fitted models and forecasts fixed",
                "input_sha256": {str(path): file_hash(path) for path in (
                    args.prepared, args.fixed, args.support / "finance_expert_labels.csv",
                    args.support / "nested_tuning_predictions.csv", args.support / "nested_tuning_metrics.csv")},
                "output_sha256": {path.name: file_hash(path) for path in sorted(args.output.glob("*.csv"))}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metric_checks": "passed", "leave_one_out": summaries,
                      "target_subperiods": period_rows}, indent=2, ensure_ascii=False))


def prepare_annotator(args: argparse.Namespace) -> None:
    blind_path, private_path = args.blind_output.resolve(), args.private_output.resolve()
    if (blind_path == private_path or blind_path in private_path.parents or
            private_path in blind_path.parents):
        raise ValueError("Blind and private directories must be separate and non-nested")
    features, _ = prepared_records(args.prepared)
    labels = read_csv(args.support / "finance_expert_labels.csv")
    sources = unique_rows(read_csv(args.support / "tcmb_statement_source_index.csv"), "origin_id")
    selected = [row for row in labels if args.split == "all" or row["split"] == args.split]
    if not selected:
        raise ValueError("No annotation documents selected")
    random.Random(args.seed).shuffle(selected)
    args.blind_output.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)
    documents, key, template = [], [], []
    fields = ["item_id"] + [field for axis in AXES for field in (axis + "_score", axis + "_evidence")]
    for row in selected:
        feature = features[row["origin_id"]]
        if (feature["decision_text_sha256"] != sources[row["origin_id"]]["decision_text_sha256"] or
                feature["origin_date"] != row["origin_date"]):
            raise ValueError("Annotation statement differs from frozen source")
        opaque = "B-" + hashlib.sha256((str(args.seed) + ":" + row["origin_id"]).encode()).hexdigest()[:12]
        documents.append({"item_id": opaque, "text": feature["decision_text"]})
        key.append({"item_id": opaque, "origin_id": row["origin_id"], "origin_date": row["origin_date"],
                    "split": row["split"], "decision_text_sha256": feature["decision_text_sha256"]})
        template.append({field: opaque if field == "item_id" else "" for field in fields})
    write_csv(args.private_output / "key.csv", key)
    write_csv(args.blind_output / "answers_template.csv", template, fields)
    (args.blind_output / "rubric.md").write_text(RUBRIC, encoding="utf-8")
    payload = json.dumps(documents, ensure_ascii=False).replace("<", "\\u003c")
    (args.blind_output / "etiketleme.html").write_text(HTML_TEMPLATE.replace("__DOCUMENTS__", payload), encoding="utf-8")
    (args.blind_output / "OKU_BENI.txt").write_text(
        "etiketleme.html dosyasını tarayıcıda açın. İnternet ve Python gerekmez.\n"
        "İlerlemeyi JSON olarak kaydet düğmesiyle ara kayıt alın; JSON yükle ile devam edin.\n"
        "Yanıtlar otomatik kaydedilmez. Bitince Tamamlanmış CSV indir düğmesine basın.\n"
        "İlk uzman etiketlerini, model sonuçlarını veya makaleyi incelemeyin.\n"
        "İndirilen ikinci_uzman_etiketleri.csv dosyasını yazarlara geri gönderin.\n", encoding="utf-8")
    manifest = {"status": "prepared only; no second-expert labels collected",
                "n_documents": len(documents), "split": args.split, "seed": args.seed,
                "prepared_sha256": file_hash(args.prepared),
                "reference_labels_sha256": file_hash(args.support / "finance_expert_labels.csv"),
                "blinding": "no original IDs, split, dates, reference labels, targets or model outputs added; original statement content retained",
                "private_key_sha256": file_hash(args.private_output / "key.csv"),
                "html_sha256": file_hash(args.blind_output / "etiketleme.html")}
    (args.private_output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {len(documents)} documents. Share ONLY {args.blind_output}.")


def kappa_scores(first: list[int], second: list[int]) -> dict:
    if len(first) != len(second) or not first or set(first + second) - {-1, 0, 1}:
        raise ValueError("Kappa requires nonempty paired -1/0/1 labels")
    matrix = np.zeros((3, 3), dtype=int)
    for original, repeated in zip(first, second):
        matrix[original + 1, repeated + 1] += 1
    observed = matrix / len(first)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    distances = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :]) / 2
    raw, chance = float(np.trace(observed)), float(np.trace(expected))
    expected_distance = float(np.sum(expected * distances))
    return {"paired_n": len(first), "agreement": raw,
            "cohen_kappa": (raw - chance) / (1 - chance) if 1 - chance > 1e-12 else None,
            "linear_weighted_kappa": 1 - float(np.sum(observed * distances)) / expected_distance
            if expected_distance > 1e-12 else None, "confusion_matrix": matrix.tolist()}


def agreement(args: argparse.Namespace) -> None:
    manifest_path = args.key.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    integrity = {"private_key": verify_csv_identity(args.key, manifest["private_key_sha256"]),
                 "first_expert_labels": verify_csv_identity(
                     args.support / "finance_expert_labels.csv", manifest["reference_labels_sha256"])}
    key = unique_rows(read_csv(args.key), "item_id")
    answers = unique_rows(read_csv(args.answers), "item_id")
    first = unique_rows(read_csv(args.support / "finance_expert_labels.csv"), "origin_id")
    if not key or set(key) != set(answers):
        raise ValueError("Answers must match every opaque ID in the private key exactly")
    result, discrepancies = [], []
    ordered = sorted(key, key=lambda identity: key[identity]["origin_date"])
    for axis in AXES:
        original, repeated = [], []
        uncertain_first = uncertain_second = 0
        for identity in ordered:
            origin = key[identity]["origin_id"]
            first_value = first[origin][axis + "_score"].strip()
            second_value = answers[identity][axis + "_score"].strip()
            if second_value not in {"-1", "0", "1", "U"}:
                raise ValueError(f"Missing or invalid label: {identity}, {axis}")
            if second_value == "U" and not answers[identity].get(axis + "_evidence", "").strip():
                raise ValueError(f"U requires a reason: {identity}, {axis}")
            uncertain_first += first_value == "U"
            uncertain_second += second_value == "U"
            if first_value != "U" and second_value != "U":
                original.append(int(first_value))
                repeated.append(int(second_value))
            if first_value != second_value:
                discrepancies.append({"item_id": identity, "origin_id": origin, "axis": axis,
                                      "first_label": first_value, "second_label": second_value})
        scores = kappa_scores(original, repeated) if original else {
            "paired_n": 0, "agreement": None, "cohen_kappa": None,
            "linear_weighted_kappa": None, "confusion_matrix": [[0] * 3 for _ in range(3)]}
        valid_pairs = scores["paired_n"]
        matrix = np.asarray(scores["confusion_matrix"])
        result.append({"axis": axis, "documents": len(ordered), "uncertain_first": uncertain_first,
                       "uncertain_second": uncertain_second, "agree_n": int(np.trace(matrix)),
                       "disagree_n": valid_pairs - int(np.trace(matrix)),
                       "first_minus_one": int(matrix.sum(axis=1)[0]),
                       "first_zero": int(matrix.sum(axis=1)[1]), "first_plus_one": int(matrix.sum(axis=1)[2]),
                       "second_minus_one": int(matrix.sum(axis=0)[0]),
                       "second_zero": int(matrix.sum(axis=0)[1]), "second_plus_one": int(matrix.sum(axis=0)[2]),
                       **scores})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "inter_annotator_agreement.csv", [
        {field: value for field, value in row.items() if field != "confusion_matrix"} for row in result])
    write_csv(args.output / "disagreements_private.csv", discrepancies,
              ["item_id", "origin_id", "axis", "first_label", "second_label"])
    report = {"status": "submitted second-expert labels compared before adjudication; no model refitting",
              "input_integrity": integrity,
              "class_order": [-1, 0, 1], "weighted_kappa": "linear ordinal distance; secondary",
              "uncertainty_rule": "exclude U only on affected axis; count separately",
              "coverage": sorted({row["split"] for row in key.values()}),
              "limitations": "No confidence intervals or full-cohort reliability claim; evaluation of models against the new reference is a separate pending step.",
              "input_sha256": {str(path): file_hash(path) for path in (
                  args.key, args.answers, args.support / "finance_expert_labels.csv")}, "axes": result}
    (args.output / "agreement_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


HTML_TEMPLATE = r'''<!doctype html>
<html lang="tr"><meta charset="utf-8"><title>Bağımsız uzman etiketleme</title>
<style>body{font:17px system-ui;max-width:950px;margin:30px auto;padding:0 16px}article{border-top:2px solid #555;padding:20px 0}pre{white-space:pre-wrap;font:inherit;line-height:1.6}label{display:block;margin:14px 0}select,textarea,button{font:inherit;padding:6px}textarea{width:95%;height:65px}header{background:#f2f5f7;padding:18px}button{margin:8px}</style>
<header><h1>Bağımsız uzman değerlendirmesi</h1>
<p>Yalnız metni değerlendirin; ilk uzman etiketlerine, model sonuçlarına veya sonraki kararlara bakmayın.</p>
<ul><li>Enflasyon: -1 azalıyor/zayıflıyor; 0 karışık, dengeli veya belirtilmemiş; +1 artıyor/güçleniyor.</li>
<li>Talep: -1 zayıf/zayıflıyor; 0 karışık, dengeli veya belirtilmemiş; +1 güçlü/güçleniyor.</li>
<li>İleri politika: -1 açık geleceğe yönelik gevşeme; 0 yönsüz/belirsiz veya yalnız mevcut sıkılığın korunması; +1 açık geleceğe yönelik ek sıkılaşma.</li></ul>
<p>Cari faiz kararı tek başına ileri politika etiketi değildir. Belirtilmemiş içerik 0'dır. Kararsızlıkta U seçin ve gerekçe yazın. Mümkünse kısa kanıt ekleyin.</p>
<p>Yanıtlar otomatik kaydedilmez. Tarayıcıyı kapatmadan önce JSON olarak kaydedin.</p>
<button id="save">İlerlemeyi JSON olarak kaydet</button><button id="export">Tamamlanmış CSV indir</button><label>JSON yükle: <input type="file" id="import" accept=".json"></label></header>
<main id="documents"></main><script>
const documents = __DOCUMENTS__;
const axes = ['inflation_pressure','demand_pressure','forward_policy_bias'];
const titles = ['Enflasyon baskısı','Talep baskısı','İleri politika eğilimi'];
const fields = ['item_id',...axes.flatMap(axis=>[axis+'_score',axis+'_evidence'])];
const formRows = [];
for(const documentItem of documents){
  const article=document.createElement('article');
  const heading=document.createElement('h2');heading.textContent=documentItem.item_id;article.append(heading);
  const text=document.createElement('pre');text.textContent=documentItem.text;article.append(text);
  const controls={};
  axes.forEach((axis,index)=>{
    const label=document.createElement('label');label.textContent=titles[index]+': ';
    const select=document.createElement('select');
    [['','Seçiniz'],['-1','-1'],['0','0'],['1','+1'],['U','U — Kararsız']].forEach(([value,title])=>{
      const option=document.createElement('option');option.value=value;option.textContent=title;select.append(option);
    });
    const evidence=document.createElement('textarea');evidence.placeholder='Metinsel kanıt veya kararsızlık gerekçesi';
    label.append(select);article.append(label,evidence);controls[axis+'_score']=select;controls[axis+'_evidence']=evidence;
  });
  formRows.push({item_id:documentItem.item_id,controls});document.getElementById('documents').append(article);
}
function values(){return formRows.map(row=>Object.fromEntries(fields.map(field=>[field,field==='item_id'?row.item_id:row.controls[field].value])));}
function download(payload,name,type){
  const objectUrl=URL.createObjectURL(new Blob([payload],{type}));
  const link=document.createElement('a');link.href=objectUrl;link.download=name;link.click();
  setTimeout(()=>URL.revokeObjectURL(objectUrl),1000);
}
document.getElementById('save').onclick=()=>download(JSON.stringify(values(),null,2),'uzman_ilerleme.json','application/json');
document.getElementById('export').onclick=()=>{
  const answers=values();
  if(answers.some(row=>axes.some(axis=>!row[axis+'_score']||(row[axis+'_score']==='U'&&!row[axis+'_evidence'].trim())))){
    alert('Her ekseni seçin; U için gerekçe yazın.');return;
  }
  const escapeCsv=value=>'"'+String(value).replaceAll('"','""')+'"';
  const records=[fields,...answers.map(row=>fields.map(field=>row[field]))];
  download('\uFEFF'+records.map(record=>record.map(escapeCsv).join(',')).join('\r\n'),'ikinci_uzman_etiketleri.csv','text/csv;charset=utf-8');
};
document.getElementById('import').onchange=async event=>{
  try{
    const selectedFile=event.target.files[0];if(!selectedFile)return;
    const answers=JSON.parse(await selectedFile.text());
    if(!Array.isArray(answers)||answers.length!==formRows.length)throw Error('Belge sayısı uyumsuz');
    const entries=new Map(answers.map(row=>[row.item_id,row]));
    if(entries.size!==formRows.length||formRows.some(row=>!entries.has(row.item_id)))throw Error('Belge kimlikleri uyumsuz');
    for(const row of answers){for(const field of fields){if(typeof row[field]!=='string')throw Error('Geçersiz kayıt');}
      for(const axis of axes){if(!['','-1','0','1','U'].includes(row[axis+'_score']))throw Error('Geçersiz etiket');}}
    for(const row of formRows){const answer=entries.get(row.item_id);fields.slice(1).forEach(field=>{row.controls[field].value=answer[field];});}
  }catch(error){alert(error.message);}
};
</script></html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "prepare-annotator", "agreement"))
    parser.add_argument("--support", type=Path, default=Path(__file__).parent)
    parser.add_argument("--prepared", type=Path, default=Path("prism-uploads/tcmb_next_meeting_prepare_v1.zip"))
    parser.add_argument("--fixed", type=Path, default=Path("prism-uploads/publication_run_results.zip"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--blind-output", type=Path, default=Path("Expert_Systems/second_expert_blind"))
    parser.add_argument("--private-output", type=Path, default=Path("Expert_Systems/second_expert_private"))
    parser.add_argument("--split", choices=("test", "all"), default="test")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--key", type=Path, default=Path("Expert_Systems/second_expert_private/key.csv"))
    parser.add_argument("--answers", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.support / "criticism_3_audit" if args.mode == "audit" else Path("Expert_Systems/second_expert_private/agreement")
    if args.mode == "agreement" and args.answers is None:
        parser.error("agreement requires --answers")
    return args


def main() -> None:
    args = parse_args()
    {"audit": audit, "prepare-annotator": prepare_annotator, "agreement": agreement}[args.mode](args)


if __name__ == "__main__":
    main()