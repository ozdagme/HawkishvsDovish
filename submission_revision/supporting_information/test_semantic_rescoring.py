import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np


SPEC = importlib.util.spec_from_file_location("rescore_semantic_predictions", Path(__file__).with_name("rescore_semantic_predictions.py"))
RESCORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESCORE)


class SemanticRescoringTests(unittest.TestCase):
    def prediction(self):
        return dict.fromkeys(RESCORE.AXES, 0)

    def test_fenced_object_is_content_valid_not_strict(self):
        prediction, strict = RESCORE.first_object("```json\n" + json.dumps(self.prediction()) + "\n```\nExplanation")
        self.assertEqual(prediction, self.prediction())
        self.assertFalse(strict)

    def test_plain_object_is_strict(self):
        _, strict = RESCORE.first_object(" \n" + json.dumps(self.prediction()) + "\n")
        self.assertTrue(strict)

    def test_duplicate_keys_are_rejected(self):
        raw = json.dumps(self.prediction()).replace('"inflation_pressure": 0', '"inflation_pressure": 0, "inflation_pressure": 1')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            RESCORE.first_object(raw)

    def test_values_are_never_coerced(self):
        for value in (True, 0.0, "0", None, 2):
            with self.subTest(value=value):
                prediction = {**self.prediction(), "inflation_pressure": value}
                with self.assertRaises(ValueError):
                    RESCORE.first_object(json.dumps(prediction))

    def test_missing_axis_is_not_repaired(self):
        with self.assertRaises(ValueError):
            RESCORE.first_object('{"inflation_pressure": 0}')

    def test_invalid_first_object_does_not_select_later_answer(self):
        with self.assertRaises(ValueError):
            RESCORE.first_object('{invalid}\n' + json.dumps(self.prediction()))

    def test_absent_classes_remain_in_macro_denominator(self):
        result = RESCORE.score(np.zeros((1, 24, 3), dtype=int), np.zeros((24, 3), dtype=int))
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["axis_mean_accuracy"], 1)

    def test_uncertain_reference_excludes_only_affected_axis(self):
        truth = np.zeros((24, 3), dtype=int)
        truth[0, 1] = RESCORE.MISSING
        runs = np.zeros((1, 24, 3), dtype=int)
        runs[0, 0, 1] = 1
        result = RESCORE.score(runs, truth)
        self.assertEqual(result["axis_n"], [24, 23, 24])
        self.assertEqual(result["axis_mean_accuracy"], 1)

    def test_seed_metrics_are_averaged_not_predictions(self):
        truth = np.zeros((24, 3), dtype=int)
        runs = np.asarray([np.zeros((24, 3), dtype=int), np.ones((24, 3), dtype=int)])
        result = RESCORE.score(runs, truth)
        self.assertAlmostEqual(result["macro_f1"], 1 / 6)
        self.assertEqual(result["axis_mean_accuracy"], 0.5)

    def test_vector_bootstrap_matches_direct_resampling(self):
        rng = np.random.default_rng(13)
        truth = rng.integers(-1, 2, size=(24, 3))
        truth[0, 1] = RESCORE.MISSING
        runs = rng.integers(-1, 2, size=(3, 24, 3))
        draws = rng.integers(0, 24, size=(10, 24))
        vector = RESCORE.bootstrap_scores(runs, truth, draws)
        direct = [RESCORE.score(runs[:, indices, :], truth[indices])["macro_f1"] for indices in draws]
        np.testing.assert_allclose(vector, direct, atol=1e-15)

    def test_changed_archive_member_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("report.json", "changed")
                archive.writestr("export_manifest.json", json.dumps({"files_sha256": {"report.json": "invalid"}}))
            with self.assertRaisesRegex(ValueError, "Export hash mismatch"):
                RESCORE.load_archive(path)

    @unittest.skipUnless(Path("prism-uploads/SEMANTIK_KAYITLI_TAHMINLER.zip").is_file(), "Requires privately retained frozen archive")
    def test_real_frozen_archive_reproduces_published_results(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            with patch("sys.argv", ["rescore_semantic_predictions.py", "--output", directory]):
                with contextlib.redirect_stdout(io.StringIO()):
                    RESCORE.main()
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(manifest["llm_records"], 672)
            self.assertEqual(manifest["first_objects_valid"], 672)
            self.assertEqual(manifest["reference_axis_n"]["second_full"], [24, 24, 24])
            self.assertEqual(manifest["reference_axis_n"]["second_common"], [24, 23, 24])


if __name__ == "__main__":
    unittest.main()