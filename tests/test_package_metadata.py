import unittest
import importlib.util

import model


class TestPackageMetadata(unittest.TestCase):
    def test_model_package_exposes_version(self):
        self.assertIsInstance(model.__version__, str)
        self.assertTrue(model.__version__)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_model_can_be_constructed(self):
        from model import MultiScaleGRFNet

        model = MultiScaleGRFNet(input_dim=12)
        self.assertGreater(sum(param.numel() for param in model.parameters()), 0)
