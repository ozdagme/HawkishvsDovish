#!/usr/bin/env python3
"""Post-hoc rolling-origin tuning for reconstructable forecasting arms.

The outer 37-origin evaluation period is never used to select hyperparameters.
For each outer origin, candidate settings are ranked only by earlier
rolling-origin predictions whose outcomes were already public. Direction and
magnitude settings are selected separately because macro-F1 and MAE are
different targets. The frozen BERT arm is included when its validated
document-embedding cache is supplied.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcmb_repro.pipeline import (  # noqa: E402
    CLASSES,
    classification_metrics,
    fit_predict,
    impute_train,
    load_dataset,
    load_embeddings,
    lsa_fit,
    qwen_designs,
    read_json,
    scale_train,
    tfidf_fit,
)

LAMBDAS = (0.01, 0.1, 1.0)
TEXT_SCALINGS = ("row_only", "column_standardized")
QWEN_ARMS = (
    "history_qwen_all",
    "history_qwen_inflation",
    "history_qwen_demand",
    "history_qwen_forward",
    "history_qwen_scores_only",
    "history_qwen_missing_only",
)
ARMS = (
    "history",
    "history_tfidf",
    "history_lsa",
    *QWEN_ARMS,
    "proxy_market",
    "proxy_market_lsa",
)
PROXY_FIELDS = (
    "cds_monthly_feature",
    "usd_buying_feature",
    "usd_selling_feature",
    "inflation_current_month_feature",
    "inflation_year_end_feature",
    "inflation_12m_feature",
    "cds_daily_feature",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--config", default="configs/publication.json")
    parser.add_argument("--embeddings", default="data/derived/bert_embeddings.npz")
    parser.add_argument("--output-dir", default="outputs/nested_tuning")
    parser.add_argument("--minimum-inner-origins", type=int, default=20)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def csv_from_zip(path: Path, member: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(member).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(raw)))


def load_market_proxy(raw_dir: Path, dataset: Any) -> np.ndarray:
    rows = csv_from_zip(raw_dir / "data.zip", "data/processed/model_dataset_enriched.csv")
    by_document = {row["document_id"]: row for row in rows}
    if len(by_document) != len(rows):
        raise ValueError("Duplicate document_id in market-proxy source")
    matrix: list[list[float]] = []
    for feature in dataset.features:
        source = by_document[feature["decision_document_id"]]
        if source["monthly_feature_rule"] != "previous_completed_month":
            raise ValueError("Unexpected monthly proxy timing rule")
        if source["daily_feature_rule"] != "strictly_previous_trading_day":
            raise ValueError("Unexpected daily proxy timing rule")
        values = [float(source[field]) if source[field].strip() else math.nan for field in PROXY_FIELDS]
        matrix.append(values)
    return np.asarray(matrix, dtype=float)


def candidate_id(arm: str, regularization: float, scaling: str = "fixed") -> str:
    return f"{arm}|lambda={regularization:g}|scaling={scaling}"


def candidate_metadata(arm: str, regularization: float, scaling: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id(arm, regularization, scaling),
        "arm": arm,
        "lambda": regularization,
        "scaling": scaling,
    }


def chronological_indices(dataset: Any, query_index: int) -> list[int]:
    origin_date = dataset.features[query_index]["origin_date"]
    return [
        index
        for index in range(query_index)
        if dataset.targets[index]["target_publication_date"] <= origin_date
    ]


def build_designs(
    dataset: Any,
    market: np.ndarray,
    embeddings: np.ndarray | None,
    query_index: int,
    train_indices: list[int],
    config: dict[str, Any],
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    history_train, history_query, _ = scale_train(
        dataset.history[train_indices], dataset.history[query_index : query_index + 1]
    )
    tfidf_train, tfidf_query, _ = tfidf_fit(
        [dataset.features[index]["decision_text"] for index in train_indices],
        dataset.features[query_index]["decision_text"],
        int(config["tfidf"]["min_df"]),
        int(config["tfidf"]["max_features"]),
    )
    lsa_train, lsa_query, _ = lsa_fit(
        tfidf_train,
        tfidf_query,
        int(config["lsa"]["components"]),
        float(config["lsa"]["rank_threshold"]),
    )
    tfidf_standardized_train, tfidf_standardized_query, _ = scale_train(
        tfidf_train, tfidf_query
    )
    lsa_standardized_train, lsa_standardized_query, _ = scale_train(
        lsa_train, lsa_query
    )
    qwen_parts, _ = qwen_designs(dataset.qwen, train_indices, query_index)
    market_train, market_query, _ = impute_train(
        market[train_indices], market[query_index : query_index + 1], include_missing=True
    )

    designs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {
        ("history", "fixed"): (history_train, history_query),
        ("history_tfidf", "row_only"): (
            np.hstack([history_train, tfidf_train]),
            np.hstack([history_query, tfidf_query]),
        ),
        ("history_tfidf", "column_standardized"): (
            np.hstack([history_train, tfidf_standardized_train]),
            np.hstack([history_query, tfidf_standardized_query]),
        ),
        ("history_lsa", "row_only"): (
            np.hstack([history_train, lsa_train]),
            np.hstack([history_query, lsa_query]),
        ),
        ("history_lsa", "column_standardized"): (
            np.hstack([history_train, lsa_standardized_train]),
            np.hstack([history_query, lsa_standardized_query]),
        ),
        ("proxy_market", "fixed"): (
            np.hstack([history_train, market_train]),
            np.hstack([history_query, market_query]),
        ),
        ("proxy_market_lsa", "row_only"): (
            np.hstack([history_train, market_train, lsa_train]),
            np.hstack([history_query, market_query, lsa_query]),
        ),
        ("proxy_market_lsa", "column_standardized"): (
            np.hstack([history_train, market_train, lsa_standardized_train]),
            np.hstack([history_query, market_query, lsa_standardized_query]),
        ),
    }
    if embeddings is not None:
        embedding_train, embedding_query, _ = scale_train(
            embeddings[train_indices], embeddings[query_index : query_index + 1]
        )
        designs[("history_transformer", "fixed")] = (
            np.hstack([history_train, embedding_train]),
            np.hstack([history_query, embedding_query]),
        )
    for arm, (part_train, part_query) in qwen_parts.items():
        designs[(arm, "fixed")] = (
            np.hstack([history_train, part_train]),
            np.hstack([history_query, part_query]),
        )
    return designs


def score_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    truth = np.asarray([CLASSES.index(row["true_class"]) for row in rows])
    predicted = np.asarray([CLASSES.index(row["predicted_class"]) for row in rows])
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=float)
    true_action = np.asarray([row["true_action_bp"] for row in rows], dtype=float)
    predicted_action = np.asarray([row["predicted_action_bp"] for row in rows], dtype=float)
    classification = classification_metrics(truth, predicted)
    return {
        "macro_f1": float(classification["macro_f1"]),
        "brier": float(np.mean(np.sum((probabilities - np.eye(3)[truth]) ** 2, axis=1))),
        "mae_bp": float(np.mean(np.abs(true_action - predicted_action))),
    }


def select_candidates(
    rows_by_candidate: dict[str, list[dict[str, Any]]],
    candidate_info: dict[str, dict[str, Any]],
    eligible_queries: set[int],
    arm: str,
    minimum_inner_origins: int,
) -> tuple[str, str, int, list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    for key, info in candidate_info.items():
        if info["arm"] != arm:
            continue
        rows = [row for row in rows_by_candidate[key] if row["query_index"] in eligible_queries]
        if len(rows) < minimum_inner_origins:
            continue
        scored.append({**info, **score_candidate_rows(rows), "inner_n": len(rows)})
    if not scored:
        raise ValueError(f"Insufficient inner origins for {arm}")
    direction = max(
        scored,
        key=lambda row: (
            row["macro_f1"],
            -row["brier"],
            row["lambda"],
            row["scaling"] == "row_only",
        ),
    )
    magnitude = min(
        scored,
        key=lambda row: (
            row["mae_bp"],
            -row["lambda"],
            row["scaling"] != "row_only",
        ),
    )
    return direction["candidate_id"], magnitude["candidate_id"], int(direction["inner_n"]), scored


def circular_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    values: list[int] = []
    while len(values) < n:
        start = int(rng.integers(0, n))
        values.extend((start + offset) % n for offset in range(block_length))
    return np.asarray(values[:n], dtype=int)


def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = np.asarray([CLASSES.index(row["true_class"]) for row in rows])
    predicted = np.asarray([CLASSES.index(row["predicted_class"]) for row in rows])
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=float)
    true_action = np.asarray([row["true_action_bp"] for row in rows], dtype=float)
    predicted_action = np.asarray([row["predicted_action_bp"] for row in rows], dtype=float)
    classification = classification_metrics(truth, predicted)
    return {
        "n": len(rows),
        "correct": int(classification["correct"]),
        "macro_f1": float(classification["macro_f1"]),
        "mae_bp": float(np.mean(np.abs(true_action - predicted_action))),
        "brier_multiclass": float(
            np.mean(np.sum((probabilities - np.eye(3)[truth]) ** 2, axis=1))
        ),
    }


def paired_contrasts(
    selected: dict[str, list[dict[str, Any]]],
    replicates: int,
    block_length: int,
    seed: int,
) -> list[dict[str, Any]]:
    baseline = selected["history"]
    n = len(baseline)
    rng = np.random.default_rng(seed)
    indices = [circular_indices(n, block_length, rng) for _ in range(replicates)]
    output: list[dict[str, Any]] = []
    for arm, rows in selected.items():
        if arm == "history":
            continue
        point_base = metrics_for_rows(baseline)
        point_arm = metrics_for_rows(rows)
        macro_draws: list[float] = []
        mae_draws: list[float] = []
        for sample in indices:
            sampled_base = [baseline[index] for index in sample]
            sampled_arm = [rows[index] for index in sample]
            base_metrics = metrics_for_rows(sampled_base)
            arm_metrics = metrics_for_rows(sampled_arm)
            macro_draws.append(arm_metrics["macro_f1"] - base_metrics["macro_f1"])
            mae_draws.append(arm_metrics["mae_bp"] - base_metrics["mae_bp"])
        output.append(
            {
                "candidate": arm,
                "macro_f1_difference": point_arm["macro_f1"] - point_base["macro_f1"],
                "macro_f1_ci_low": float(np.quantile(macro_draws, 0.025)),
                "macro_f1_ci_high": float(np.quantile(macro_draws, 0.975)),
                "mae_bp_difference": point_arm["mae_bp"] - point_base["mae_bp"],
                "mae_bp_ci_low": float(np.quantile(mae_draws, 0.025)),
                "mae_bp_ci_high": float(np.quantile(mae_draws, 0.975)),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    raw_dir = resolve(args.raw_dir)
    config = read_json(resolve(args.config))
    output_dir = resolve(args.output_dir)
    dataset = load_dataset(raw_dir, config["evaluation_origin_start"])
    market = load_market_proxy(raw_dir, dataset)
    embeddings_path = resolve(args.embeddings)
    embeddings: np.ndarray | None = None
    embedding_metadata: dict[str, Any] | None = None
    if embeddings_path.is_file():
        embeddings, embedding_metadata = load_embeddings(embeddings_path, dataset)
        if embedding_metadata.get("backend") != "transformer":
            raise ValueError("Nested publication analysis requires transformer embeddings")
        expected_commit = config["embedding"].get("revision")
        if expected_commit and embedding_metadata.get("resolved_commit") != expected_commit:
            raise ValueError(
                "Embedding model commit does not match configs/publication.json: "
                f"{embedding_metadata.get('resolved_commit')} != {expected_commit}"
            )
    arms = (*ARMS, "history_transformer") if embeddings is not None else ARMS

    rows_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_info: dict[str, dict[str, Any]] = {}
    eligible_query_indices: list[int] = []

    for query_index in range(len(dataset.features)):
        train_indices = chronological_indices(dataset, query_index)
        labels = np.asarray(
            [CLASSES.index(dataset.targets[index]["direction"]) for index in train_indices]
        )
        if len(train_indices) < 40 or set(labels.tolist()) != {0, 1, 2}:
            continue
        eligible_query_indices.append(query_index)
        actions = np.asarray(
            [dataset.targets[index]["action_bp"] / 100.0 for index in train_indices]
        )
        designs = build_designs(
            dataset, market, embeddings, query_index, train_indices, config
        )
        truth = dataset.targets[query_index]
        for (arm, scaling), (train, query) in designs.items():
            for regularization in LAMBDAS:
                run_config = copy.deepcopy(config)
                run_config["regularization_lambda"] = regularization
                estimate, _ = fit_predict(train, labels, actions, query, run_config)
                info = candidate_metadata(arm, regularization, scaling)
                candidate_info[info["candidate_id"]] = info
                rows_by_candidate[info["candidate_id"]].append(
                    {
                        "query_index": query_index,
                        "origin_id": dataset.features[query_index]["origin_id"],
                        "origin_date": dataset.features[query_index]["origin_date"],
                        "true_class": truth["direction"],
                        "true_action_bp": truth["action_bp"],
                        **estimate,
                    }
                )
        print(
            json.dumps(
                {
                    "query_index": query_index,
                    "origin_date": dataset.features[query_index]["origin_date"],
                    "eligible_queries_completed": len(eligible_query_indices),
                }
            ),
            flush=True,
        )

    by_query_candidate = {
        (row["query_index"], key): row
        for key, rows in rows_by_candidate.items()
        for row in rows
    }
    selected: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    choice_rows: list[dict[str, Any]] = []
    inner_score_rows: list[dict[str, Any]] = []
    outer_indices = {int(fold["i"]) for fold in dataset.folds}
    for fold in dataset.folds:
        query_index = int(fold["i"])
        eligible_inner = set(fold["training_indices"]) & set(eligible_query_indices)
        for arm in arms:
            direction_key, magnitude_key, inner_n, scored = select_candidates(
                rows_by_candidate,
                candidate_info,
                eligible_inner,
                arm,
                args.minimum_inner_origins,
            )
            direction_row = by_query_candidate[(query_index, direction_key)]
            magnitude_row = by_query_candidate[(query_index, magnitude_key)]
            selected_row = {
                "fold": int(fold["fold"]),
                "origin_id": fold["origin_id"],
                "origin_date": fold["origin_date"],
                "arm": arm,
                "inner_n": inner_n,
                "direction_candidate": direction_key,
                "magnitude_candidate": magnitude_key,
                "true_class": direction_row["true_class"],
                "predicted_class": direction_row["predicted_class"],
                "probabilities": json.dumps(direction_row["probabilities"]),
                "true_action_bp": direction_row["true_action_bp"],
                "predicted_action_bp": magnitude_row["predicted_action_bp"],
            }
            selected[arm].append(
                {
                    **selected_row,
                    "probabilities": direction_row["probabilities"],
                }
            )
            choice_rows.append(
                {
                    "fold": int(fold["fold"]),
                    "origin_id": fold["origin_id"],
                    "origin_date": fold["origin_date"],
                    "arm": arm,
                    "inner_n": inner_n,
                    "direction_candidate": direction_key,
                    "magnitude_candidate": magnitude_key,
                }
            )
            for row in scored:
                inner_score_rows.append(
                    {
                        "fold": int(fold["fold"]),
                        "origin_id": fold["origin_id"],
                        **row,
                    }
                )

    prediction_rows = []
    for arm in arms:
        for row in selected[arm]:
            prediction_rows.append(
                {
                    **row,
                    "probabilities": json.dumps(row["probabilities"]),
                }
            )
    metric_rows = [{"arm": arm, **metrics_for_rows(rows)} for arm, rows in selected.items()]
    contrasts = paired_contrasts(
        selected,
        int(config["bootstrap"]["replicates"]),
        int(config["bootstrap"]["block_length"]),
        20260923,
    )
    write_csv(output_dir / "predictions.csv", prediction_rows)
    write_csv(output_dir / "choices.csv", choice_rows)
    write_csv(output_dir / "inner_scores.csv", inner_score_rows)
    write_csv(output_dir / "metrics.csv", metric_rows)
    write_csv(output_dir / "contrasts_vs_tuned_history.csv", contrasts)
    metadata = {
        "status": "post-hoc nested rolling-origin analysis",
        "outer_origins": len(dataset.folds),
        "eligible_historical_tuning_origins": len(eligible_query_indices),
        "minimum_inner_origins": args.minimum_inner_origins,
        "lambdas": list(LAMBDAS),
        "text_scalings": list(TEXT_SCALINGS),
        "direction_selection": "maximum inner macro-F1; lower Brier, stronger lambda, and row-only scaling as deterministic tie-breakers",
        "magnitude_selection": "minimum inner MAE; stronger lambda and row-only scaling as deterministic tie-breakers",
        "bert_status": (
            "included from validated transformer embedding cache"
            if embeddings is not None
            else "not tuned: generate data/derived/bert_embeddings.npz and rerun"
        ),
        "embedding_path": str(embeddings_path) if embeddings is not None else None,
        "embedding_metadata": embedding_metadata,
        "outer_indices_match_publication_folds": outer_indices
        == {int(fold["i"]) for fold in dataset.folds},
        "bootstrap": {
            "replicates": int(config["bootstrap"]["replicates"]),
            "block_length": int(config["bootstrap"]["block_length"]),
            "seed": 20260923,
            "models_refitted_inside_bootstrap": False,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), **metadata}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())