from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import logsumexp


EXPECTED_INPUTS = {
    "data.zip": "26ea3b8c6632585102f9c3e2659de894a48e6f88adcb9da59c07fbadaffdb846",
    "tcmb_next_meeting_prepare_v1.zip": "d520ccfe09e40c7a6192d6aa51b6ceacb28afd3fddcf6a8e5d3af1ddd05e1f38",
    "tcmb_calendar_audit_v1_repeat2.zip": "f4f5af38ce3ad4a22e4e6b4c623cbb9cb069fa085f914fcfab83ff10ba43f905",
    "tcmb_qwen_features_v1.zip": "58067daf21bcd5f1a3d53c9cfe4c387acfeae0c1f80d5030bab997a99ce476d8",
}

CLASSES = ("cut", "hold", "hike")
QWEN_AXES = ("inflation_pressure", "demand_pressure", "forward_policy_bias")
ARM_ORDER = (
    "history",
    "history_tfidf",
    "history_lsa",
    "history_transformer",
    "history_qwen_all",
    "history_qwen_inflation",
    "history_qwen_demand",
    "history_qwen_forward",
    "history_qwen_scores_only",
    "history_qwen_missing_only",
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_from_zip(archive: zipfile.ZipFile, member: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in archive.read(member).decode("utf-8").splitlines()
        if line.strip()
    ]


def verify_inputs(raw_dir: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in EXPECTED_INPUTS.items():
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
        observed[name] = actual
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"CRC failure in {name}")
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate ZIP member in {name}")
            if any(Path(member).is_absolute() or ".." in Path(member).parts for member in names):
                raise ValueError(f"Unsafe ZIP path in {name}")
    return observed


def stage_inputs(source_dir: Path, raw_dir: Path) -> dict[str, str]:
    """Copy the four frozen archives from an existing project directory.

    Direct ZIP files and ZIP members nested in output bundles are supported.
    A file is accepted only if its SHA-256 matches the frozen research input.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    archive_paths = sorted(
        (
            path
            for path in source_dir.rglob("*.zip")
            if path.is_file() and raw_dir not in path.parents
        ),
        key=str,
    )
    archives_by_hash: dict[str, list[Path]] = {}
    for path in archive_paths:
        try:
            archives_by_hash.setdefault(sha256_file(path), []).append(path)
        except OSError:
            continue

    staged: dict[str, str] = {}
    for name, expected in EXPECTED_INPUTS.items():
        destination = raw_dir / name
        if destination.is_file() and sha256_file(destination) == expected:
            staged[name] = "already_present"
            continue

        matching = archives_by_hash.get(expected, [])
        if matching:
            source = sorted(matching, key=lambda path: (len(path.parts), str(path)))[0]
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != expected:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Copy verification failed for {name}")
            os.replace(temporary, destination)
            staged[name] = str(source)
            continue

        nested_source: str | None = None
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        for outer_path in archive_paths:
            try:
                with zipfile.ZipFile(outer_path) as outer:
                    members = [
                        info
                        for info in outer.infolist()
                        if not info.is_dir() and Path(info.filename).name == name
                    ]
                    for member in members:
                        payload = outer.read(member)
                        if hashlib.sha256(payload).hexdigest() != expected:
                            continue
                        with temporary.open("wb") as handle:
                            handle.write(payload)
                        nested_source = f"{outer_path}!{member.filename}"
                        break
            except (OSError, zipfile.BadZipFile, RuntimeError):
                continue
            if nested_source is not None:
                break

        if nested_source is None:
            temporary.unlink(missing_ok=True)
            raise FileNotFoundError(
                f"Could not find frozen {name}, directly or inside an output ZIP, "
                f"under {source_dir}. "
                f"Expected SHA-256: {expected}"
            )
        if sha256_file(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Nested extraction verification failed for {name}")
        os.replace(temporary, destination)
        staged[name] = nested_source
    verify_inputs(raw_dir)
    return staged


@dataclass(frozen=True)
class Dataset:
    features: list[dict[str, Any]]
    targets: list[dict[str, Any]]
    folds: list[dict[str, Any]]
    history: np.ndarray
    qwen: np.ndarray
    qwen_status: list[str]


def load_dataset(raw_dir: Path, evaluation_origin_start: str) -> Dataset:
    prepared_path = raw_dir / "tcmb_next_meeting_prepare_v1.zip"
    with zipfile.ZipFile(prepared_path) as archive:
        features = jsonl_from_zip(archive, "features_decision_origin.jsonl")
        targets = jsonl_from_zip(archive, "evaluation_only_targets.jsonl")

    if len(features) != 123 or len(targets) != 123:
        raise ValueError("Expected 123 prepared origins and targets")
    if targets[-1].get("label_status") != "unobserved":
        raise ValueError("The final meeting must have an unobserved next target")
    features = features[:-1]
    targets = targets[:-1]

    for index, (feature, target) in enumerate(zip(features, targets)):
        if feature["origin_id"] != target["origin_id"]:
            raise ValueError(f"Origin/target mismatch at row {index}")
        if feature["origin_date"] >= target["target_publication_date"]:
            raise ValueError(f"Non-future target at row {index}")
        expected_direction = (
            "cut" if target["action_bp"] < 0 else "hike" if target["action_bp"] > 0 else "hold"
        )
        if target["direction"] != expected_direction:
            raise ValueError(f"Direction/action mismatch at row {index}")

    qwen, qwen_status = load_qwen_features(raw_dir, features)
    history = history_features(features)
    folds: list[dict[str, Any]] = []
    for index, feature in enumerate(features):
        if feature["origin_date"] < evaluation_origin_start:
            continue
        train = [
            j
            for j in range(index)
            if targets[j]["target_publication_date"] <= feature["origin_date"]
        ]
        if not train:
            raise ValueError(f"Empty training fold for {feature['origin_id']}")
        folds.append(
            {
                "fold": len(folds) + 1,
                "i": index,
                "origin_id": feature["origin_id"],
                "origin_date": feature["origin_date"],
                "training_indices": train,
                "latest_training_target_date": max(
                    targets[j]["target_publication_date"] for j in train
                ),
            }
        )
    if len(folds) != 37:
        raise ValueError(f"Expected 37 evaluation folds, found {len(folds)}")
    return Dataset(features, targets, folds, history, qwen, qwen_status)


def validate_qwen_response(raw: str, text: str, ended_with_eos: bool) -> tuple[str, list[float] | None]:
    try:
        if ended_with_eos is not True:
            raise ValueError("missing EOS")
        parsed = json.loads(raw.strip())
        if not isinstance(parsed, dict) or set(parsed) != set(QWEN_AXES):
            raise ValueError("wrong output schema")
        values: list[float] = []
        for axis in QWEN_AXES:
            item = parsed[axis]
            if not isinstance(item, dict) or set(item) != {"score", "evidence"}:
                raise ValueError("wrong axis schema")
            score, evidence = item["score"], item["evidence"]
            if type(score) is not int or score not in (-1, 0, 1):
                raise ValueError("invalid score")
            if not isinstance(evidence, str) or len(evidence) > 240:
                raise ValueError("invalid evidence")
            if score != 0 and not evidence.strip():
                raise ValueError("non-zero score without evidence")
            if evidence and evidence not in text:
                raise ValueError("evidence is not a literal substring")
            values.append(float(score))
        return "valid", values
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return "invalid_output", None


def load_qwen_features(raw_dir: Path, features: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    archive_path = raw_dir / "tcmb_qwen_features_v1.zip"
    matrix = np.full((len(features), len(QWEN_AXES)), np.nan, dtype=float)
    statuses: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        requests = json.loads(archive.read("requests.json"))
        request_by_id = {row["request_id"]: row for row in requests}
        if len(request_by_id) != 122:
            raise ValueError("Expected 122 unique Qwen requests")
        for index, feature in enumerate(features):
            origin = feature["origin_id"]
            record = json.loads(archive.read(f"records/{origin}.json"))
            request = request_by_id[origin]
            if request["i"] != index or record["i"] != index:
                raise ValueError(f"Qwen index mismatch for {origin}")
            if request["text_sha256"] != feature["decision_text_sha256"]:
                raise ValueError(f"Qwen text mismatch for {origin}")
            status, values = validate_qwen_response(
                record["raw_response"], feature["decision_text"], record["ended_with_eos"]
            )
            if status != record["status"]:
                raise ValueError(f"Qwen validation mismatch for {origin}")
            statuses.append(status)
            if values is not None:
                matrix[index] = values
    if statuses.count("valid") != 95:
        raise ValueError("Expected 95 valid archived Qwen outputs")
    return matrix, statuses


def history_features(features: list[dict[str, Any]]) -> np.ndarray:
    rows: list[list[float]] = []
    for index, feature in enumerate(features):
        previous = features[index - 1] if index else None
        recent = features[max(0, index - 2) : index + 1]
        rows.append(
            [
                float(feature["repo_rate_after_current_decision"]),
                float(feature["current_decision_change_bp"]) / 100.0,
                float(previous["current_decision_change_bp"]) / 100.0 if previous else 0.0,
                float(np.mean([row["current_decision_change_bp"] / 100.0 for row in recent])),
                float(
                    (date.fromisoformat(feature["origin_date"]) - date.fromisoformat(previous["origin_date"])).days
                )
                if previous
                else 0.0,
                float(previous is not None),
            ]
        )
    return np.asarray(rows, dtype=float)


TOKEN_RE = re.compile(r"[a-zçğıöşü]{2,}")


def words(text: str) -> list[str]:
    return TOKEN_RE.findall(text.replace("İ", "i").replace("I", "ı").lower())


def scale_train(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError("Non-finite numeric input before scaling")
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (query - mean) / std, {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def impute_train(
    train: np.ndarray, query: np.ndarray, include_missing: bool
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    medians = np.asarray(
        [
            np.median(train[np.isfinite(train[:, column]), column])
            if np.isfinite(train[:, column]).any()
            else 0.0
            for column in range(train.shape[1])
        ]
    )

    def fill(values: np.ndarray) -> np.ndarray:
        missing = (~np.isfinite(values)).astype(float)
        filled = np.where(np.isfinite(values), values, medians)
        return np.hstack([filled, missing]) if include_missing else filled

    scaled_train, scaled_query, metadata = scale_train(fill(train), fill(query))
    metadata.update({"medians": medians.tolist(), "include_missing": include_missing})
    return scaled_train, scaled_query, metadata


def missing_only_train(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return scale_train((~np.isfinite(train)).astype(float), (~np.isfinite(query)).astype(float))


def tfidf_fit(
    texts: list[str], query: str, min_df: int, max_features: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    counts = [Counter(words(text)) for text in texts]
    document_frequency = Counter(token for count in counts for token in count)
    vocabulary = sorted(
        (token for token, count in document_frequency.items() if count >= min_df),
        key=lambda token: (-document_frequency[token], token),
    )[:max_features]
    if not vocabulary:
        raise ValueError("Empty TF-IDF vocabulary")
    idf = np.asarray(
        [math.log((1 + len(counts)) / (1 + document_frequency[token])) + 1 for token in vocabulary]
    )

    def transform(count: Counter[str]) -> np.ndarray:
        vector = np.asarray(
            [(1 + math.log(count[token])) if count[token] else 0.0 for token in vocabulary]
        ) * idf
        norm = max(float(np.linalg.norm(vector)), 1e-15)
        return vector / norm

    train = np.asarray([transform(count) for count in counts])
    transformed_query = transform(Counter(words(query)))[None, :]
    return train, transformed_query, {
        "vocabulary_size": len(vocabulary),
        "vocabulary_sha256": sha256_bytes("\n".join(vocabulary).encode("utf-8")),
    }


def lsa_fit(
    train: np.ndarray, query: np.ndarray, components: int, rank_threshold: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    eigenvalues, eigenvectors = np.linalg.eigh(train @ train.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    keep = min(
        components,
        int(np.sum(eigenvalues > max(float(eigenvalues[0]), 1.0) * rank_threshold)),
    )
    if keep <= 0:
        raise ValueError("Zero numerical LSA rank")
    directions = (eigenvectors[:, :keep].T @ train) / np.sqrt(eigenvalues[:keep])[:, None]
    for row in directions:
        if row[int(np.argmax(np.abs(row)))] < 0:
            row *= -1

    def normalize(matrix: np.ndarray) -> np.ndarray:
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-15)

    return normalize(train @ directions.T), normalize(query @ directions.T), {
        "effective_rank": keep,
        "eigenvalues": eigenvalues[:keep].tolist(),
    }


def hash_embeddings(texts: list[str], dimension: int = 64) -> np.ndarray:
    matrix = np.zeros((len(texts), dimension), dtype=np.float64)
    for row_index, text in enumerate(texts):
        tokens = words(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            matrix[row_index, index] += sign
        norm = np.linalg.norm(matrix[row_index])
        if norm:
            matrix[row_index] /= norm
    return matrix


def prepare_transformer_chunks(tokenizer: Any, chunks: list[list[int]]) -> list[dict[str, list[int]]]:
    """Prepare BERT chunks using stable token-id attributes only."""
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    if cls_token_id is None or sep_token_id is None:
        raise ValueError("The frozen BERT tokenizer must define CLS and SEP token IDs")
    prepared: list[dict[str, list[int]]] = []
    for chunk in chunks:
        input_ids = [int(cls_token_id), *chunk, int(sep_token_id)]
        prepared.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "token_type_ids": [0] * len(input_ids),
            }
        )
    return prepared


def pad_transformer_chunks(
    prepared: list[dict[str, list[int]]], pad_token_id: int
) -> dict[str, list[list[int]]]:
    """Right-pad prepared BERT chunks without tokenizer padding helpers."""
    width = max(len(item["input_ids"]) for item in prepared)
    batch = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
    for item in prepared:
        padding = width - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [pad_token_id] * padding)
        batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
        batch["token_type_ids"].append(item["token_type_ids"] + [0] * padding)
    return batch


def transformer_embeddings(
    texts: list[str], embedding_config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformer experiment requires torch and transformers. Install requirements.txt first."
        ) from exc

    model_name = embedding_config["model_name"]
    revision = embedding_config.get("revision", "main")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    model = AutoModel.from_pretrained(model_name, revision=revision)
    model.eval()
    requested_device = embedding_config.get("device", "auto")
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    model.to(device)
    max_length = min(int(embedding_config.get("max_length", 512)), int(tokenizer.model_max_length))
    overlap = int(embedding_config.get("chunk_overlap", 64))
    content_length = max_length - int(tokenizer.num_special_tokens_to_add(pair=False))
    stride = content_length - overlap
    if stride <= 0:
        raise ValueError("chunk_overlap must be smaller than content token capacity")

    document_vectors: list[np.ndarray] = []
    chunk_counts: list[int] = []
    with torch.no_grad():
        for text in texts:
            token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            if not token_ids:
                token_ids = [tokenizer.unk_token_id]
            chunks = [token_ids[start : start + content_length] for start in range(0, len(token_ids), stride)]
            chunk_vectors: list[np.ndarray] = []
            for start in range(0, len(chunks), int(embedding_config.get("batch_size", 8))):
                batch_chunks = chunks[start : start + int(embedding_config.get("batch_size", 8))]
                prepared = prepare_transformer_chunks(tokenizer, batch_chunks)
                padded = pad_transformer_chunks(prepared, int(tokenizer.pad_token_id or 0))
                encoded = {
                    key: torch.tensor(value, dtype=torch.long)
                    for key, value in padded.items()
                }
                encoded = {key: value.to(device) for key, value in encoded.items()}
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                chunk_vectors.extend(pooled.detach().cpu().numpy())
            document = np.mean(np.asarray(chunk_vectors, dtype=np.float64), axis=0)
            if embedding_config.get("l2_normalize", True):
                document /= max(float(np.linalg.norm(document)), 1e-15)
            document_vectors.append(document)
            chunk_counts.append(len(chunks))
    commit = getattr(model.config, "_commit_hash", None)
    metadata = {
        "backend": "transformer",
        "model_name": model_name,
        "requested_revision": revision,
        "resolved_commit": commit,
        "device": device,
        "torch": torch.__version__,
        "transformers_model_type": model.config.model_type,
        "hidden_size": int(model.config.hidden_size),
        "max_length": max_length,
        "chunk_overlap": overlap,
        "chunk_counts": chunk_counts,
    }
    return np.asarray(document_vectors), metadata


def save_embeddings(
    path: Path,
    origin_ids: list[str],
    text_hashes: list[str],
    matrix: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        origin_ids=np.asarray(origin_ids),
        text_hashes=np.asarray(text_hashes),
        embeddings=matrix,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )


def load_embeddings(path: Path, dataset: Dataset) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing embedding cache: {path}")
    with np.load(path, allow_pickle=False) as archive:
        origin_ids = archive["origin_ids"].tolist()
        text_hashes = archive["text_hashes"].tolist()
        matrix = archive["embeddings"].astype(float)
        metadata = json.loads(str(archive["metadata"].item()))
    expected_ids = [row["origin_id"] for row in dataset.features]
    expected_hashes = [row["decision_text_sha256"] for row in dataset.features]
    if origin_ids != expected_ids or text_hashes != expected_hashes:
        raise ValueError("Embedding cache does not match the prepared decision texts")
    if matrix.shape[0] != len(dataset.features) or not np.isfinite(matrix).all():
        raise ValueError("Invalid embedding matrix")
    return matrix, metadata


def choose_class(probabilities: np.ndarray) -> int:
    return max((1, 0, 2), key=lambda index: float(probabilities[index]))


def softmax_objective(
    flat: np.ndarray, train: np.ndarray, labels: np.ndarray, regularization: float
) -> tuple[float, np.ndarray]:
    weights = flat[:-3].reshape(train.shape[1], 3)
    intercept = flat[-3:]
    logits = train @ weights + intercept
    normalizer = logsumexp(logits, axis=1)
    value = float(
        np.mean(normalizer - logits[np.arange(len(labels)), labels])
        + regularization * np.sum(weights * weights) / 2.0
    )
    probabilities = np.exp(logits - normalizer[:, None])
    probabilities[np.arange(len(labels)), labels] -= 1.0
    gradient = np.concatenate(
        [
            (train.T @ probabilities / len(labels) + regularization * weights).ravel(),
            probabilities.mean(axis=0),
        ]
    )
    return value, gradient


def fit_predict(
    train: np.ndarray,
    labels: np.ndarray,
    actions_percentage_points: np.ndarray,
    query: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(labels.tolist()) != {0, 1, 2}:
        raise ValueError("A training fold lacks one of the three classes")
    regularization = float(config["regularization_lambda"])
    optimizer = config["optimizer"]
    result = minimize(
        softmax_objective,
        np.zeros(train.shape[1] * 3 + 3),
        args=(train, labels, regularization),
        jac=True,
        method=optimizer["method"],
        options={key: value for key, value in optimizer.items() if key != "method"},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Classifier convergence failure: {result.message}")
    weights = result.x[:-3].reshape(train.shape[1], 3)
    intercept = result.x[-3:]
    logits = query @ weights + intercept
    probabilities = np.exp(logits - logsumexp(logits, axis=1)[:, None])[0]

    feature_mean = train.mean(axis=0)
    action_mean = float(actions_percentage_points.mean())
    centered = train - feature_mean
    ridge_weights = centered.T @ np.linalg.solve(
        centered @ centered.T + len(labels) * regularization * np.eye(len(labels)),
        actions_percentage_points - action_mean,
    )
    ridge_intercept = action_mean - float(feature_mean @ ridge_weights)
    predicted_action_bp = float((query @ ridge_weights + ridge_intercept)[0] * 100.0)
    prediction = {
        "predicted_class": CLASSES[choose_class(probabilities)],
        "probabilities": probabilities.tolist(),
        "predicted_action_bp": predicted_action_bp,
    }
    model = {
        "feature_count": int(train.shape[1]),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "classification_weights_sha256": sha256_bytes(weights.tobytes()),
        "ridge_weights_sha256": sha256_bytes(ridge_weights.tobytes()),
    }
    return prediction, model


def qwen_designs(
    qwen: np.ndarray, train_indices: list[int], query_index: int
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    train_raw = qwen[train_indices]
    query_raw = qwen[query_index : query_index + 1]
    designs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metadata: dict[str, Any] = {}

    full_train, full_query, metadata["history_qwen_all"] = impute_train(
        train_raw, query_raw, include_missing=True
    )
    designs["history_qwen_all"] = (full_train, full_query)

    for column, suffix in enumerate(("inflation", "demand", "forward")):
        axis_train, axis_query, metadata[f"history_qwen_{suffix}"] = impute_train(
            train_raw[:, column : column + 1],
            query_raw[:, column : column + 1],
            include_missing=True,
        )
        designs[f"history_qwen_{suffix}"] = (axis_train, axis_query)

    scores_train, scores_query, metadata["history_qwen_scores_only"] = impute_train(
        train_raw, query_raw, include_missing=False
    )
    designs["history_qwen_scores_only"] = (scores_train, scores_query)

    missing_train_values, missing_query_values, metadata["history_qwen_missing_only"] = missing_only_train(
        train_raw, query_raw
    )
    designs["history_qwen_missing_only"] = (missing_train_values, missing_query_values)
    return designs, metadata


def run_folds(
    dataset: Dataset,
    embeddings: np.ndarray,
    embedding_metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    fold_audit: list[dict[str, Any]] = []
    tfidf_config = config["tfidf"]
    lsa_config = config["lsa"]
    for fold in dataset.folds:
        query_index = int(fold["i"])
        train_indices = list(fold["training_indices"])
        if any(index >= query_index for index in train_indices):
            raise ValueError(f"Non-past training row in fold {fold['fold']}")
        if any(
            dataset.targets[index]["target_publication_date"] > fold["origin_date"]
            for index in train_indices
        ):
            raise ValueError(f"Unannounced target in fold {fold['fold']}")

        history_train, history_query, history_meta = scale_train(
            dataset.history[train_indices], dataset.history[query_index : query_index + 1]
        )
        tfidf_train, tfidf_query, tfidf_meta = tfidf_fit(
            [dataset.features[index]["decision_text"] for index in train_indices],
            dataset.features[query_index]["decision_text"],
            min_df=int(tfidf_config["min_df"]),
            max_features=int(tfidf_config["max_features"]),
        )
        lsa_train, lsa_query, lsa_meta = lsa_fit(
            tfidf_train,
            tfidf_query,
            components=int(lsa_config["components"]),
            rank_threshold=float(lsa_config["rank_threshold"]),
        )
        embedding_train, embedding_query, embedding_scale = scale_train(
            embeddings[train_indices], embeddings[query_index : query_index + 1]
        )
        qwen_parts, qwen_meta = qwen_designs(dataset.qwen, train_indices, query_index)

        designs: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "history": (history_train, history_query),
            "history_tfidf": (
                np.hstack([history_train, tfidf_train]),
                np.hstack([history_query, tfidf_query]),
            ),
            "history_lsa": (
                np.hstack([history_train, lsa_train]),
                np.hstack([history_query, lsa_query]),
            ),
            "history_transformer": (
                np.hstack([history_train, embedding_train]),
                np.hstack([history_query, embedding_query]),
            ),
        }
        for arm, (train_part, query_part) in qwen_parts.items():
            designs[arm] = (
                np.hstack([history_train, train_part]),
                np.hstack([history_query, query_part]),
            )
        if tuple(designs) != ARM_ORDER:
            raise ValueError(f"Unexpected arm order: {tuple(designs)}")

        labels = np.asarray(
            [CLASSES.index(dataset.targets[index]["direction"]) for index in train_indices]
        )
        actions = np.asarray(
            [dataset.targets[index]["action_bp"] / 100.0 for index in train_indices]
        )
        truth = dataset.targets[query_index]
        model_audit: dict[str, Any] = {}
        for arm, (train, query) in designs.items():
            prediction, model = fit_predict(train, labels, actions, query, config)
            model_audit[arm] = model
            records.append(
                {
                    "fold": fold["fold"],
                    "origin_id": fold["origin_id"],
                    "origin_date": fold["origin_date"],
                    "training_n": len(train_indices),
                    "arm": arm,
                    "true_class": truth["direction"],
                    "true_action_bp": truth["action_bp"],
                    **prediction,
                }
            )
        fold_audit.append(
            {
                **fold,
                "transformations": {
                    "history": history_meta,
                    "tfidf": tfidf_meta,
                    "lsa": lsa_meta,
                    "transformer_scaling": embedding_scale,
                    "qwen": qwen_meta,
                },
                "models": model_audit,
                "embedding_backend": embedding_metadata["backend"],
            }
        )
    return records, fold_audit


def classification_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((3, 3), dtype=int)
    for actual, estimate in zip(truth, predicted):
        confusion[int(actual), int(estimate)] += 1
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    class_f1 = np.divide(
        2 * np.diag(confusion),
        denominator,
        out=np.zeros(3, dtype=float),
        where=denominator != 0,
    )
    return {
        "macro_f1": float(class_f1.mean()),
        "accuracy": float(np.trace(confusion) / len(truth)),
        "correct": int(np.trace(confusion)),
        "n": int(len(truth)),
        "class_f1": class_f1.tolist(),
        "confusion": confusion.tolist(),
    }


def score_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics_rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARM_ORDER:
        rows = [row for row in records if row["arm"] == arm]
        rows.sort(key=lambda row: int(row["fold"]))
        truth = np.asarray([CLASSES.index(row["true_class"]) for row in rows])
        predicted = np.asarray([CLASSES.index(row["predicted_class"]) for row in rows])
        probabilities = np.asarray([row["probabilities"] for row in rows], dtype=float)
        true_action = np.asarray([row["true_action_bp"] for row in rows], dtype=float)
        predicted_action = np.asarray([row["predicted_action_bp"] for row in rows], dtype=float)
        classification = classification_metrics(truth, predicted)
        metric = {
            "arm": arm,
            **classification,
            "mae_bp": float(np.mean(np.abs(true_action - predicted_action))),
            "brier_multiclass": float(
                np.mean(np.sum((probabilities - np.eye(3)[truth]) ** 2, axis=1))
            ),
            "log_loss": float(
                -np.mean(np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-15, 1.0)))
            ),
        }
        metrics_rows.append(metric)
        arrays[arm] = {
            "truth": truth,
            "predicted": predicted,
            "true_action": true_action,
            "predicted_action": predicted_action,
        }
    return metrics_rows, arrays


def paired_block_contrasts(
    metrics_rows: list[dict[str, Any]],
    arrays: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    bootstrap = config["bootstrap"]
    rng = np.random.Generator(np.random.PCG64(int(bootstrap["seed"])))
    n = len(arrays["history"]["truth"])
    block_length = int(bootstrap["block_length"])
    repeats = int(bootstrap["replicates"])
    shared_indices = np.empty((repeats, n), dtype=np.int32)
    for repeat in range(repeats):
        starts = rng.integers(0, n, size=(n + block_length - 1) // block_length)
        shared_indices[repeat] = (
            (starts[:, None] + np.arange(block_length)[None, :]) % n
        ).ravel()[:n]
    metric_by_arm = {row["arm"]: row for row in metrics_rows}
    contrasts: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        if arm == "history":
            continue
        macro_draws = np.empty(repeats)
        mae_draws = np.empty(repeats)
        for repeat in range(repeats):
            indices = shared_indices[repeat]
            baseline = arrays["history"]
            candidate = arrays[arm]
            truth = baseline["truth"][indices]
            macro_draws[repeat] = (
                classification_metrics(truth, candidate["predicted"][indices])["macro_f1"]
                - classification_metrics(truth, baseline["predicted"][indices])["macro_f1"]
            )
            mae_draws[repeat] = float(
                np.mean(
                    np.abs(candidate["true_action"][indices] - candidate["predicted_action"][indices])
                )
                - np.mean(
                    np.abs(baseline["true_action"][indices] - baseline["predicted_action"][indices])
                )
            )
        contrasts.append(
            {
                "candidate": arm,
                "reference": "history",
                "macro_f1_difference": metric_by_arm[arm]["macro_f1"]
                - metric_by_arm["history"]["macro_f1"],
                "macro_f1_ci_low": float(np.quantile(macro_draws, 0.025, method="linear")),
                "macro_f1_ci_high": float(np.quantile(macro_draws, 0.975, method="linear")),
                "mae_bp_difference": metric_by_arm[arm]["mae_bp"]
                - metric_by_arm["history"]["mae_bp"],
                "mae_bp_ci_low": float(np.quantile(mae_draws, 0.025, method="linear")),
                "mae_bp_ci_high": float(np.quantile(mae_draws, 0.975, method="linear")),
                "bootstrap_replicates": repeats,
                "block_length": block_length,
            }
        )
    return contrasts


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(normalized)


def summary_markdown(
    metrics_rows: list[dict[str, Any]], contrasts: list[dict[str, Any]], metadata: dict[str, Any]
) -> str:
    lines = [
        "# TCMB financial-NLP extension results",
        "",
        "**Status:** Post-hoc extension on previously observed dates; not an independent confirmatory test.",
        "",
        f"Embedding backend: `{metadata['backend']}`",
        "",
        "| Arm | Correct/N | Macro-F1 | MAE (bp) | Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['arm']} | {row['correct']}/{row['n']} | {row['macro_f1']:.4f} | "
            f"{row['mae_bp']:.2f} | {row['brier_multiclass']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired differences relative to history",
            "",
            "| Candidate | Macro-F1 difference [95% interval] | MAE difference [95% interval] |",
            "|---|---:|---:|",
        ]
    )
    for row in contrasts:
        lines.append(
            f"| {row['candidate']} | {row['macro_f1_difference']:.4f} "
            f"[{row['macro_f1_ci_low']:.4f}, {row['macro_f1_ci_high']:.4f}] | "
            f"{row['mae_bp_difference']:.2f} "
            f"[{row['mae_bp_ci_low']:.2f}, {row['mae_bp_ci_high']:.2f}] |"
        )
    lines.extend(
        [
            "",
            "Hash embeddings, if present in a smoke-test directory, are software checks only and must not be reported.",
            "",
        ]
    )
    return "\n".join(lines)


def create_output_manifest(output_dir: Path, run_metadata: dict[str, Any]) -> None:
    files: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(output_dir))] = sha256_file(path)
    write_json(output_dir / "manifest.json", {"run": run_metadata, "files": files})


def run_experiment(
    raw_dir: Path,
    output_dir: Path,
    embeddings_path: Path,
    config: dict[str, Any],
    allow_nonpublication_backend: bool = False,
) -> dict[str, Any]:
    input_hashes = verify_inputs(raw_dir)
    dataset = load_dataset(raw_dir, config["evaluation_origin_start"])
    embeddings, embedding_metadata = load_embeddings(embeddings_path, dataset)
    if embedding_metadata.get("backend") != "transformer" and not allow_nonpublication_backend:
        raise ValueError("Non-transformer embeddings are smoke-test only")
    records, fold_audit = run_folds(dataset, embeddings, embedding_metadata, config)
    metrics_rows, arrays = score_records(records)
    contrasts = paired_block_contrasts(metrics_rows, arrays, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "predictions.csv",
        records,
        [
            "fold",
            "origin_id",
            "origin_date",
            "training_n",
            "arm",
            "true_class",
            "predicted_class",
            "true_action_bp",
            "predicted_action_bp",
            "probabilities",
        ],
    )
    write_csv(output_dir / "metrics.csv", metrics_rows)
    write_csv(output_dir / "contrasts_vs_history.csv", contrasts)
    write_json(output_dir / "fold_audit.json", fold_audit)
    run_metadata = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "protocol_status": config["status"],
        "config_sha256": sha256_bytes(canonical_json(config)),
        "input_sha256": input_hashes,
        "embedding": embedding_metadata,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "folds": len(dataset.folds),
        "origins": len(dataset.features),
        "qwen_valid": dataset.qwen_status.count("valid"),
        "qwen_invalid": dataset.qwen_status.count("invalid_output"),
        "new_llm_calls": 0,
    }
    write_json(output_dir / "run_metadata.json", run_metadata)
    (output_dir / "RESULTS.md").write_text(
        summary_markdown(metrics_rows, contrasts, embedding_metadata), encoding="utf-8"
    )
    create_output_manifest(output_dir, run_metadata)
    return run_metadata


def build_embeddings(
    raw_dir: Path,
    output_path: Path,
    config: dict[str, Any],
    backend: str | None = None,
) -> dict[str, Any]:
    verify_inputs(raw_dir)
    dataset = load_dataset(raw_dir, config["evaluation_origin_start"])
    selected_backend = backend or config["embedding"]["backend"]
    texts = [row["decision_text"] for row in dataset.features]
    if selected_backend == "transformer":
        matrix, metadata = transformer_embeddings(texts, config["embedding"])
    elif selected_backend == "hash":
        matrix = hash_embeddings(texts)
        metadata = {
            "backend": "hash",
            "dimension": int(matrix.shape[1]),
            "publication_eligible": False,
            "purpose": "deterministic smoke test only",
        }
    else:
        raise ValueError(f"Unknown embedding backend: {selected_backend}")
    metadata["created_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["matrix_sha256"] = sha256_bytes(matrix.tobytes())
    expected_matrix_sha256 = config.get("embedding", {}).get("expected_matrix_sha256")
    if selected_backend == "transformer" and expected_matrix_sha256:
        metadata["expected_matrix_sha256"] = expected_matrix_sha256
        metadata["matches_expected_matrix_sha256"] = (
            metadata["matrix_sha256"] == expected_matrix_sha256
        )
    save_embeddings(
        output_path,
        [row["origin_id"] for row in dataset.features],
        [row["decision_text_sha256"] for row in dataset.features],
        matrix,
        metadata,
    )
    return metadata


def check_repository(raw_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    hashes = verify_inputs(raw_dir)
    dataset = load_dataset(raw_dir, config["evaluation_origin_start"])
    return {
        "status": "ok",
        "input_sha256": hashes,
        "origins": len(dataset.features),
        "folds": len(dataset.folds),
        "first_evaluation_origin": dataset.folds[0]["origin_date"],
        "last_evaluation_origin": dataset.folds[-1]["origin_date"],
        "qwen_valid": dataset.qwen_status.count("valid"),
        "qwen_invalid": dataset.qwen_status.count("invalid_output"),
    }