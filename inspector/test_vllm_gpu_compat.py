"""vLLM GPU image selection vs host compute capability."""

from __future__ import annotations

import os
import unittest
from datetime import timedelta
from unittest import mock

import lib


class VllmGpuCompatTest(unittest.TestCase):
    def setUp(self) -> None:
        self._cap = os.environ.get("GPU_COMPUTE_CAP")
        os.environ.pop("GPU_COMPUTE_CAP", None)

    def tearDown(self) -> None:
        os.environ.pop("GPU_COMPUTE_CAP", None)
        if self._cap is not None:
            os.environ["GPU_COMPUTE_CAP"] = self._cap

    def test_parse_compute_cap(self) -> None:
        self.assertEqual(lib._parse_compute_cap("7.5"), (7, 5))
        self.assertEqual(lib._parse_compute_cap(" 8.9\n"), (8, 9))
        self.assertIsNone(lib._parse_compute_cap("T4"))

    def test_env_caps(self) -> None:
        os.environ["GPU_COMPUTE_CAP"] = "7.5,7.5"
        self.assertEqual(lib._gpu_compute_capabilities(), [(7, 5), (7, 5)])

    def test_skip_gpu_image_on_turing(self) -> None:
        os.environ["GPU_COMPUTE_CAP"] = "7.5"
        task = lib.VllmDockerTask(
            images=[
                "ghcr.io/sparecores/benchmark-vllm-gpu:main",
                "ghcr.io/sparecores/benchmark-vllm-cpu:main",
            ],
            command=None,
            timeout=timedelta(hours=1),
        )
        attempts = lib._vllm_image_attempts(task, gpu_count=2)
        self.assertEqual(
            [label for label, _, _ in attempts],
            ["benchmark-vllm-cpu"],
        )

    def test_keep_gpu_image_on_ampere(self) -> None:
        os.environ["GPU_COMPUTE_CAP"] = "8.9"
        task = lib.VllmDockerTask(
            images=[
                "ghcr.io/sparecores/benchmark-vllm-gpu:main",
                "ghcr.io/sparecores/benchmark-vllm-cpu:main",
            ],
            command=None,
            timeout=timedelta(hours=1),
        )
        attempts = lib._vllm_image_attempts(task, gpu_count=1)
        self.assertEqual(
            [label for label, _, _ in attempts],
            ["benchmark-vllm-gpu", "benchmark-vllm-cpu"],
        )

    def test_unknown_caps_still_try_gpu(self) -> None:
        with mock.patch.object(lib, "_gpu_compute_capabilities", return_value=None):
            task = lib.VllmDockerTask(
                images=["ghcr.io/sparecores/benchmark-vllm-gpu:main"],
                command=None,
                timeout=timedelta(hours=1),
            )
            attempts = lib._vllm_image_attempts(task, gpu_count=1)
        self.assertEqual([label for label, _, _ in attempts], ["benchmark-vllm-gpu"])


if __name__ == "__main__":
    unittest.main()
