import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark import main, run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_cpu_run_returns_real_measurements(self):
        report = run_benchmark(size=16, runs=3, warmup=1, device="cpu")
        self.assertEqual(report["backend"], "cpu")
        self.assertEqual(len(report["samples_ms"]), 3)
        self.assertTrue(all(math.isfinite(x) and x > 0 for x in report["samples_ms"]))
        self.assertGreaterEqual(report["mean_ms"], report["min_ms"])
        self.assertIsNone(report["peak_allocated_mib"])

    def test_invalid_settings_are_rejected(self):
        for settings in ({"size": 0}, {"runs": 0}, {"warmup": -1},
                         {"dtype": "int8"}, {"device": "mps"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                run_benchmark(**settings)

    def test_explicit_gpu_request_does_not_silently_use_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                run_benchmark(size=16, device="cuda")

    def test_auto_uses_cpu_when_gpu_is_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            report = run_benchmark(size=16, runs=1, warmup=0, device="auto")
        self.assertEqual(report["device"], "cpu")

    def test_cli_writes_json_and_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "results" / "cpu.json"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(["--device", "cpu", "--size", "16", "--runs", "2",
                      "--warmup", "0", "--output", str(destination)])
            report = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(output.getvalue()), report)
            self.assertEqual(report["runs"], 2)

    def test_cli_reports_invalid_settings(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["--size", "0"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
