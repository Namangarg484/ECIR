"""Mocked runtime configuration checks; does not allocate GPU tensors."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .run import configure_runtime


class RuntimeTests(unittest.TestCase):
    def args(self, **changes):
        return SimpleNamespace(**dict(dict(device="mps", cpu_threads=4,
                                          interop_threads=1, determinism="warn"), **changes))

    def test_rejects_invalid_threads(self):
        with self.assertRaises(ValueError):
            configure_runtime(self.args(cpu_threads=0))

    def test_unavailable_mps_does_not_fall_back(self):
        with patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "MPS requested but unavailable"):
                configure_runtime(self.args())

    def test_mps_threads_and_explicit_warning_policy(self):
        with patch("torch.backends.mps.is_available", return_value=True), \
                patch("torch.set_num_threads") as threads, \
                patch("torch.set_num_interop_threads") as interop, \
                patch("torch.use_deterministic_algorithms") as deterministic:
            configure_runtime(self.args())
            threads.assert_called_once_with(4)
            interop.assert_called_once_with(1)
            deterministic.assert_called_once_with(True, warn_only=True)


if __name__ == "__main__":
    unittest.main()
