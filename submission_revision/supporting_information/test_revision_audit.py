import importlib.util
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np


SPEC = importlib.util.spec_from_file_location("revision_audit", Path(__file__).with_name("revision_audit.py"))
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
EXPORT_SPEC = importlib.util.spec_from_file_location("export_semantic_predictions", Path(__file__).with_name("export_semantic_predictions.py"))
EXPORT = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(EXPORT)


class RevisionAuditTests(unittest.TestCase):
    def test_majority_macro_f1(self):
        truth = np.asarray(["cut"] * 9 + ["hold"] * 18 + ["hike"] * 10)
        predicted = np.asarray(["hold"] * 37)
        self.assertAlmostEqual(AUDIT.macro_f1(truth, predicted), 0.21818181818181817)

    def test_absent_class_ceiling(self):
        truth = np.asarray(["hold", "hike"])
        self.assertAlmostEqual(AUDIT.macro_f1(truth, truth), 2 / 3)

    def test_duplicate_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            AUDIT.unique_rows([{"item_id": "one"}, {"item_id": "one"}], "item_id")

    def test_kappa_perfect_agreement(self):
        result = AUDIT.kappa_scores([-1, 0, 1], [-1, 0, 1])
        self.assertEqual(result["agreement"], 1)
        self.assertEqual(result["cohen_kappa"], 1)
        self.assertEqual(result["linear_weighted_kappa"], 1)

    def test_kappa_degenerate_marginals(self):
        result = AUDIT.kappa_scores([0, 0], [0, 0])
        self.assertEqual(result["agreement"], 1)
        self.assertIsNone(result["cohen_kappa"])
        self.assertIsNone(result["linear_weighted_kappa"])

    def test_kappa_disagreement(self):
        result = AUDIT.kappa_scores([-1, 0, 1], [1, 0, -1])
        self.assertAlmostEqual(result["agreement"], 1 / 3)
        self.assertAlmostEqual(result["cohen_kappa"], 0)
        self.assertAlmostEqual(result["linear_weighted_kappa"], -0.5)

    def test_unmatched_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            AUDIT.pair_rows([{"arm": "base", "origin_id": "one"},
                             {"arm": "candidate", "origin_id": "two"}], "base", "candidate")

    def test_invalid_probability_is_rejected(self):
        row = {"true_class": "hold", "predicted_class": "hold", "probabilities": "[0, 2, 0]",
               "true_action_bp": "0", "predicted_action_bp": "0"}
        with self.assertRaises(ValueError):
            AUDIT.metrics([row])

    def test_pair_with_different_target_is_rejected(self):
        baseline = {"arm": "base", "origin_id": "one", "origin_date": "2023-01-19",
                    "true_class": "hold", "true_action_bp": "0"}
        candidate = {**baseline, "arm": "candidate", "true_class": "hike"}
        with self.assertRaises(ValueError):
            AUDIT.pair_rows([baseline, candidate], "base", "candidate")

    def test_unobserved_terminal_target_is_accepted(self):
        feature = {"origin_id": "last", "origin_date": "2026-07-23", "decision_text": "metin",
                   "decision_text_sha256": hashlib.sha256("metin".encode()).hexdigest()}
        target = {"origin_id": "last", "target_publication_date": None, "label_status": "unobserved"}
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "prepared.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("features_decision_origin.jsonl", json.dumps(feature) + "\n")
                archive.writestr("evaluation_only_targets.jsonl", json.dumps(target) + "\n")
            features, targets = AUDIT.prepared_records(archive_path)
            self.assertEqual(features["last"]["decision_text"], "metin")
            self.assertIsNone(targets["last"]["target_publication_date"])

    def test_blind_and_private_directories_cannot_overlap(self):
        args = argparse.Namespace(blind_output=Path("packet"), private_output=Path("packet/private"))
        with self.assertRaises(ValueError):
            AUDIT.prepare_annotator(args)

    def test_csv_identity_accepts_only_equivalent_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key.csv"
            AUDIT.write_csv(path, [{"item_id": "B-one", "origin_id": "origin-one"}])
            expected = AUDIT.file_hash(path)
            text = path.read_text(encoding="utf-8-sig").removesuffix("\n")
            path.write_text(text, encoding="utf-8")
            result = AUDIT.verify_csv_identity(path, expected)
            self.assertEqual(result["status"], "csv_serialization_equivalent")
            path.write_text(text.replace("origin-one", "origin-two"), encoding="utf-8")
            with self.assertRaises(ValueError):
                AUDIT.verify_csv_identity(path, expected)

    def agreement_fixture(self, directory):
        root = Path(directory)
        support = root / "support"
        key_rows, labels, answers = [], [], []
        for index, score in enumerate((-1, 0, 1)):
            key_rows.append({"item_id": f"B-{index}", "origin_id": f"origin-{index}",
                             "origin_date": f"2024-0{index + 1}-01", "split": "test"})
            labels.append({"origin_id": f"origin-{index}", **{axis + "_score": str(score) for axis in AUDIT.AXES}})
            answers.append({"item_id": f"B-{index}", **{axis + "_score": str(score) for axis in AUDIT.AXES},
                            **{axis + "_evidence": "" for axis in AUDIT.AXES}})
        key_path, answers_path = root / "key.csv", root / "answers.csv"
        AUDIT.write_csv(key_path, key_rows)
        AUDIT.write_csv(support / "finance_expert_labels.csv", labels)
        AUDIT.write_csv(answers_path, answers)
        (root / "manifest.json").write_text(json.dumps({
            "private_key_sha256": AUDIT.file_hash(key_path),
            "reference_labels_sha256": AUDIT.file_hash(support / "finance_expert_labels.csv")}), encoding="utf-8")
        return argparse.Namespace(key=key_path, answers=answers_path, support=support, output=root / "result"), answers

    def test_agreement_end_to_end_excludes_only_uncertain_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            args, answers = self.agreement_fixture(directory)
            answers[0]["demand_pressure_score"] = "U"
            answers[0]["demand_pressure_evidence"] = "uncertain synthetic fixture"
            AUDIT.write_csv(args.answers, answers)
            with contextlib.redirect_stdout(io.StringIO()):
                AUDIT.agreement(args)
            report = json.loads((args.output / "agreement_report.json").read_text())
            self.assertEqual([row["paired_n"] for row in report["axes"]], [3, 2, 3])
            self.assertEqual(report["axes"][1]["uncertain_second"], 1)
            self.assertEqual(report["axes"][0]["cohen_kappa"], 1)

    def test_agreement_rejects_incomplete_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            args, answers = self.agreement_fixture(directory)
            answers[0]["forward_policy_bias_score"] = ""
            AUDIT.write_csv(args.answers, answers)
            with self.assertRaises(ValueError):
                AUDIT.agreement(args)

    def test_agreement_rejects_changed_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.agreement_fixture(directory)
            labels = AUDIT.read_csv(args.support / "finance_expert_labels.csv")
            labels[0]["inflation_pressure_score"] = "0"
            AUDIT.write_csv(args.support / "finance_expert_labels.csv", labels)
            with self.assertRaises(ValueError):
                AUDIT.agreement(args)


class SemanticExportTests(unittest.TestCase):
    def prepare_run(self, directory):
        root = Path(directory) / "run"
        root.mkdir(parents=True)
        documents = [{"id": f"doc-{index}", "split": "validation" if index < 18 else "test"}
                     for index in range(42)]
        (root / "split_manifest.json").write_text(json.dumps({"documents": documents}))
        for family in ("qwen", "mistral"):
            for seed in (None, 20260916, 20260917, 20260918):
                variant = "base" if seed is None else f"ft_{seed}"
                for rag in ("no_rag", "rag"):
                    for document in documents:
                        task = f"predictions/{family}/{variant}/{rag}/{document['id']}.json"
                        record = {**document, "family": family, "seed": seed, "rag": rag,
                                  "prediction": {axis: 0 for axis in AUDIT.AXES}}
                        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True,
                                             allow_nan=False, indent=2).encode()
                        path = root / task
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps({"identity": "synthetic-test-fixture", "task": task,
                                                    "sha": hashlib.sha256(encoded).hexdigest(), "data": record}))
        for name in ["report.json", "results.csv", "axis_results.csv", "generation_summary.csv",
                     "tasks/cpu.json", "tasks/bert.json"]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        return root

    def call_export(self, root, output):
        root = root.relative_to(Path.cwd()) if root.is_absolute() else root
        output = output.relative_to(Path.cwd()) if output.is_absolute() else output
        with patch("sys.argv", ["export_semantic_predictions.py", "--run-dir", str(root), "--output", str(output)]):
            with contextlib.redirect_stdout(io.StringIO()):
                EXPORT.main()

    def test_complete_export_omits_weights(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = self.prepare_run(directory)
            output = Path(directory) / "export.zip"
            self.call_export(root, output)
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("export_manifest.json"))
                self.assertEqual(manifest["llm_records"], 672)
                self.assertFalse(manifest["model_weights_included"])
                self.assertTrue(all(not name.startswith("models/") for name in archive.namelist()))

    def test_corrupted_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = self.prepare_run(directory)
            path = next((root / "predictions").rglob("*.json"))
            envelope = json.loads(path.read_text())
            envelope["data"]["prediction"]["forward_policy_bias"] = 1
            path.write_text(json.dumps(envelope))
            with self.assertRaises(ValueError):
                self.call_export(root, Path(directory) / "export.zip")

    def test_discovery_reports_empty_run_without_export(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory) / "empty"
            root.mkdir()
            documents = [{"id": f"doc-{index}", "split": "validation" if index < 18 else "test"}
                         for index in range(42)]
            (root / "split_manifest.json").write_text(json.dumps({"documents": documents}))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                ready = EXPORT.discover_runs(Path(directory))
            self.assertEqual(ready, [])
            self.assertIn("0/672", output.getvalue())
            self.assertIn("predictions directory is missing", output.getvalue())
            self.assertFalse(list(Path(directory).rglob("*.zip")))

    def test_discovery_finds_valid_run_and_rejects_corrupt_run(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            valid = self.prepare_run(Path(directory) / "valid")
            corrupt = self.prepare_run(Path(directory) / "corrupt")
            path = next((corrupt / "predictions").rglob("*.json"))
            envelope = json.loads(path.read_text())
            envelope["sha"] = "invalid"
            path.write_text(json.dumps(envelope))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                ready = EXPORT.discover_runs(Path(directory))
            self.assertEqual(ready, [valid])
            self.assertIn("Corrupted stored prediction", output.getvalue())

    def call_auto(self, directory, output):
        directory = Path(directory).relative_to(Path.cwd())
        output = output.relative_to(Path.cwd())
        with patch("sys.argv", ["export_semantic_predictions.py", "--auto", "--search-root",
                                str(directory), "--output", str(output)]):
            with contextlib.redirect_stdout(io.StringIO()):
                EXPORT.main()

    def test_auto_exports_one_valid_run(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            self.prepare_run(directory)
            output = Path(directory) / "export.zip"
            self.call_auto(directory, output)
            self.assertTrue(output.is_file())

    def test_auto_rejects_multiple_valid_runs(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            self.prepare_run(Path(directory) / "first")
            self.prepare_run(Path(directory) / "second")
            output = Path(directory) / "export.zip"
            with self.assertRaisesRegex(ValueError, "exactly one validated run; found 2"):
                self.call_auto(directory, output)
            self.assertFalse(output.exists())

    def test_auto_rejects_no_valid_run(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            output = Path(directory) / "export.zip"
            with self.assertRaisesRegex(ValueError, "exactly one validated run; found 0"):
                self.call_auto(directory, output)
            self.assertFalse(output.exists())

    def test_incomplete_prediction_run_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = self.prepare_run(directory)
            next((root / "predictions").rglob("*.json")).unlink()
            with self.assertRaises(ValueError):
                self.call_export(root, Path(directory) / "export.zip")


if __name__ == "__main__":
    unittest.main()