from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcmb_repro.pipeline import (  # noqa: E402
    EXPECTED_INPUTS,
    hash_embeddings,
    load_dataset,
    pad_transformer_chunks,
    prepare_transformer_chunks,
    read_json,
    stage_inputs,
    verify_inputs,
    words,
)


class PipelineUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = ROOT / "data/raw"
        cls.config = read_json(ROOT / "configs/publication.json")

    def test_input_hashes(self) -> None:
        self.assertEqual(verify_inputs(self.raw), EXPECTED_INPUTS)

    def test_frozen_cohort(self) -> None:
        dataset = load_dataset(self.raw, self.config["evaluation_origin_start"])
        self.assertEqual(len(dataset.features), 122)
        self.assertEqual(len(dataset.folds), 37)
        self.assertEqual(dataset.folds[0]["origin_date"], "2023-01-19")
        self.assertEqual(dataset.folds[-1]["origin_date"], "2026-06-11")
        self.assertEqual(dataset.qwen_status.count("valid"), 95)
        self.assertEqual(dataset.qwen_status.count("invalid_output"), 27)

    def test_hash_embeddings_are_deterministic(self) -> None:
        texts = ["Para politikası sıkıdır.", "Talep görünümü dengelidir."]
        first = hash_embeddings(texts)
        second = hash_embeddings(texts)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 64))

    def test_turkish_tokenisation(self) -> None:
        self.assertEqual(words("İç talep ve ENFLASYON güçlü."), ["iç", "talep", "ve", "enflasyon", "güçlü"])

    def test_transformer_chunk_preparation_without_deprecated_api(self) -> None:
        class TokenizerStub:
            cls_token_id = 101
            sep_token_id = 102

        prepared = prepare_transformer_chunks(TokenizerStub(), [[11, 12], [21]])
        self.assertEqual(prepared[0]["input_ids"], [101, 11, 12, 102])
        self.assertEqual(prepared[0]["token_type_ids"], [0, 0, 0, 0])
        self.assertEqual(prepared[1]["input_ids"], [101, 21, 102])
        padded = pad_transformer_chunks(prepared, pad_token_id=0)
        self.assertEqual(padded["input_ids"], [[101, 11, 12, 102], [101, 21, 102, 0]])
        self.assertEqual(padded["attention_mask"], [[1, 1, 1, 1], [1, 1, 1, 0]])

    def test_stage_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "raw"
            report = stage_inputs(self.raw, destination)
            self.assertEqual(set(report), set(EXPECTED_INPUTS))
            self.assertEqual(verify_inputs(destination), EXPECTED_INPUTS)

    def test_stage_inputs_from_nested_output_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source = temporary / "outputs"
            destination = temporary / "raw"
            source.mkdir()
            with zipfile.ZipFile(source / "archived_run_bundle.zip", "w") as bundle:
                for name in EXPECTED_INPUTS:
                    bundle.write(self.raw / name, arcname=f"frozen_inputs/{name}")
            report = stage_inputs(source, destination)
            self.assertTrue(all("!frozen_inputs/" in value for value in report.values()))
            self.assertEqual(verify_inputs(destination), EXPECTED_INPUTS)


class PipelineIntegrationTest(unittest.TestCase):
    def test_complete_smoke_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            embedding = temporary / "smoke.npz"
            output = temporary / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run.py"),
                    "embed",
                    "--backend",
                    "hash",
                    "--embeddings",
                    str(embedding),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run.py"),
                    "run",
                    "--embeddings",
                    str(embedding),
                    "--output",
                    str(output),
                    "--force",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            # Publication mode must reject smoke-only embeddings.
            self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()