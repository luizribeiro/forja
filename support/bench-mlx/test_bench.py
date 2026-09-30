import argparse
import itertools
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import bench
import compare
import matmul
import quantized_matmul


class SuiteTests(unittest.TestCase):
    def test_matmul_shape_classes_cover_one_decode_token(self) -> None:
        self.assertEqual(sum(shape[3] for shape in matmul.SHAPES), 197)

    def test_quantized_shapes_cover_target_projections(self) -> None:
        self.assertEqual(
            quantized_matmul.SHAPES,
            [
                ("q", 2048, 4096, 4),
                ("k/v", 2048, 512, 4),
                ("o", 4096, 2048, 4),
                ("expert gate/up", 2048, 768, 4),
                ("expert down", 768, 2048, 4),
                ("router", 2048, 128, 8),
            ],
        )

    def test_main_uses_the_shared_suite_values(self) -> None:
        class FakeArray:
            dtype = "bf16"

        class FakeModel:
            @staticmethod
            def parameters() -> dict[str, FakeArray]:
                return {"weight": FakeArray()}

        class FakeTokenizer:
            pass

        response = types.SimpleNamespace(prompt_tps=256.0)
        core = types.ModuleType("mlx.core")
        core.__version__ = "0"
        core.array = FakeArray
        core.bfloat16 = "bf16"
        core.floating = object()
        core.device_info = lambda: {"device_name": "stub"}
        core.eval = lambda _values: None
        core.issubdtype = lambda _dtype, _kind: True
        mlx = types.ModuleType("mlx")
        mlx.__path__ = []
        mlx.core = core
        utils = types.ModuleType("mlx.utils")
        utils.tree_flatten = lambda _values: [("weight", FakeArray())]
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.load = lambda *_args, **_kwargs: (
            FakeModel(),
            FakeTokenizer(),
            {"vocab_size": 128},
        )
        mlx_lm.stream_generate = lambda *_args, max_tokens, **_kwargs: iter(
            [response] * max_tokens
        )

        suite = Path(__file__).resolve().parents[2] / "bench/suites/default.toml"
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            model_dir = Path(directory) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")
            arguments = argparse.Namespace(
                model_dir=model_dir,
                suite=suite,
                json=report_path,
            )
            modules = {"mlx": mlx, "mlx.core": core, "mlx.utils": utils, "mlx_lm": mlx_lm}
            clock = (value / 100 for value in itertools.count())
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(bench, "arguments", return_value=arguments),
                mock.patch.object(bench, "command_output", return_value="deadbeef"),
                mock.patch.object(bench.importlib.metadata, "version", return_value="0"),
                mock.patch.object(bench.platform, "mac_ver", return_value=("26.6", (), "")),
                mock.patch.object(bench.time, "perf_counter", side_effect=clock),
            ):
                bench.main()
            report = json.loads(report_path.read_text())
            self.assertEqual(report["settings"]["pp"], 512)
            self.assertEqual(report["settings"]["tg"], 128)
            self.assertEqual(report["tg_context_start"], 9)

    def test_reads_the_committed_default_suite(self) -> None:
        path = Path(__file__).resolve().parents[2] / "bench/suites/default.toml"
        suite = bench.load_suite(path)
        self.assertEqual(suite["pp"], 512)
        self.assertEqual(suite["decode_prefill"], 8)
        self.assertEqual(suite["contexts"], [9, 512, 2048, 4000])

    def test_supplies_the_transformers_olmoe_norm_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text('{"model_type":"olmoe"}')
            self.assertEqual(bench.model_overrides(model), {"rms_norm_eps": 1e-5})

    def test_rejects_non_increasing_contexts(self) -> None:
        source = """[bench]
pp = 512
tg = 128
reps = 30
warmups = 3
decode_prefill = 8
contexts = [9, 9]
selection = ["gpu-sequential"]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.toml"
            path.write_text(source)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                bench.load_suite(path)

    def test_compare_reads_both_record_schemas(self) -> None:
        summary = {"tokens_per_second": {"wall": {"median": 1.0, "ci95": [0.5, 1.5]}}}
        first = {
            "schema_version": 1,
            "settings": {"decode_prefill": 8},
            "tg_context_start": 9,
            "results": [{"provenance": {"engine": "old"}, "prompt_processing": summary, "token_generation": summary}],
        }
        second = {
            "schema_version": 2,
            "config": "[bench]\ndecode_prefill = 8\n",
            "inputs": [{"engine_sha256": "abcdef0123456789"}],
            "results": [{"input": 0, "pp": summary, "tg": summary}],
        }
        self.assertEqual(compare.settings(first), compare.settings(second))
        self.assertEqual(compare.engine(second, second["results"][0]), "abcdef012345")
        self.assertEqual(compare.rate(second, second["results"][0], "pp"), "1.00 (0.50–1.50)")


if __name__ == "__main__":
    unittest.main()
