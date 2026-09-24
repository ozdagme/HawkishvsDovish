#!/usr/bin/env python3
"""Post-hoc scaling and regularisation sensitivity for classical text arms."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcmb_repro.pipeline import (  # noqa: E402
    CLASSES,
    classification_metrics,
    fit_predict,
    load_dataset,
    lsa_fit,
    read_json,
    scale_train,
    tfidf_fit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--config", default="configs/publication.json")
    parser.add_argument(
        "--output", default="outputs/regularization_sensitivity.csv"
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    config = read_json(resolve(args.config))
    dataset = load_dataset(resolve(args.raw_dir), config["evaluation_origin_start"])
    output_rows: list[dict[str, object]] = []

    for scaling in ("row_only", "column_standardized"):
        for regularization in (0.001, 0.01, 0.1, 1.0, 10.0):
            run_config = copy.deepcopy(config)
            run_config["regularization_lambda"] = regularization
            predictions: dict[str, list[dict[str, object]]] = {
                "history": [],
                "history_tfidf": [],
                "history_lsa": [],
            }

            for fold in dataset.folds:
                query_index = int(fold["i"])
                train_indices = list(fold["training_indices"])
                history_train, history_query, _ = scale_train(
                    dataset.history[train_indices],
                    dataset.history[query_index : query_index + 1],
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
                if scaling == "column_standardized":
                    tfidf_train, tfidf_query, _ = scale_train(
                        tfidf_train, tfidf_query
                    )
                    lsa_train, lsa_query, _ = scale_train(lsa_train, lsa_query)

                designs = {
                    "history": (history_train, history_query),
                    "history_tfidf": (
                        np.hstack([history_train, tfidf_train]),
                        np.hstack([history_query, tfidf_query]),
                    ),
                    "history_lsa": (
                        np.hstack([history_train, lsa_train]),
                        np.hstack([history_query, lsa_query]),
                    ),
                }
                labels = np.asarray(
                    [
                        CLASSES.index(dataset.targets[index]["direction"])
                        for index in train_indices
                    ]
                )
                actions = np.asarray(
                    [dataset.targets[index]["action_bp"] / 100.0 for index in train_indices]
                )
                truth = dataset.targets[query_index]
                for arm, (train, query) in designs.items():
                    estimate, _ = fit_predict(
                        train, labels, actions, query, run_config
                    )
                    predictions[arm].append(
                        {
                            "true_class": truth["direction"],
                            "true_action_bp": truth["action_bp"],
                            **estimate,
                        }
                    )

            for arm, rows in predictions.items():
                truth = np.asarray(
                    [CLASSES.index(str(row["true_class"])) for row in rows]
                )
                predicted = np.asarray(
                    [CLASSES.index(str(row["predicted_class"])) for row in rows]
                )
                probabilities = np.asarray(
                    [row["probabilities"] for row in rows], dtype=float
                )
                true_action = np.asarray(
                    [row["true_action_bp"] for row in rows], dtype=float
                )
                predicted_action = np.asarray(
                    [row["predicted_action_bp"] for row in rows], dtype=float
                )
                classification = classification_metrics(truth, predicted)
                output_rows.append(
                    {
                        "text_scaling": scaling,
                        "lambda": regularization,
                        "arm": arm,
                        "correct": classification["correct"],
                        "n": len(rows),
                        "macro_f1": classification["macro_f1"],
                        "mae_bp": float(np.mean(np.abs(true_action - predicted_action))),
                        "brier_multiclass": float(
                            np.mean(
                                np.sum(
                                    (probabilities - np.eye(3)[truth]) ** 2, axis=1
                                )
                            )
                        ),
                    }
                )

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({"status": "ok", "rows": len(output_rows), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())