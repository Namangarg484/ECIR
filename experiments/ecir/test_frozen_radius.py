import unittest

import torch

from experiments.ecir.frozen_radius import FrozenCenterRadius
from src.models.vce_model import VCEModel


class FrozenRadiusTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = VCEModel(embed_dim=8, num_heads=4, dropout=0.0)
        self.base.eval()
        self.query = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
        self.anchors = torch.nn.functional.normalize(torch.randn(3, 4, 8), dim=-1)
        self.mask = torch.tensor([[False, False, False, False],
                                  [False, False, True, True],
                                  [False, False, False, True]])

    def test_hard_clip_reproduces_legacy_forward(self):
        expected_center, expected_kappa, _ = self.base(
            self.query, self.anchors, self.mask)
        model = FrozenCenterRadius(self.base, "hard_clip")
        center, kappa = model(self.query, self.anchors, self.mask)
        torch.testing.assert_close(center, expected_center)
        torch.testing.assert_close(kappa, expected_kappa)

    def test_smooth_parameterization_is_bounded_and_differentiable(self):
        model = FrozenCenterRadius(self.base, "smooth")
        center, features = model.center_and_features(
            self.query, self.anchors, self.mask)
        self.assertEqual(center.shape, self.query.shape)
        kappa = model.kappa_from_features(features)
        self.assertTrue(bool(torch.all(kappa > 10.0)))
        self.assertTrue(bool(torch.all(kappa < 500.0)))
        kappa.sum().backward()
        gradients = [parameter.grad for parameter in model.radius_parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_radius_update_does_not_change_center(self):
        model = FrozenCenterRadius(self.base, "smooth")
        original = model.center_digest()
        optimizer = torch.optim.SGD(list(model.radius_parameters()), lr=0.01)
        _, kappa = model(self.query, self.anchors, self.mask)
        optimizer.zero_grad()
        kappa.mean().backward()
        optimizer.step()
        model.assert_center_frozen()
        self.assertEqual(model.center_digest(), original)


if __name__ == "__main__":
    unittest.main()
