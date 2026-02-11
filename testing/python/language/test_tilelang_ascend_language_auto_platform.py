import unittest
from unittest.mock import MagicMock, patch
import sys
from tilelang.utils.target import determine_platform


class TestAutoPlatform(unittest.TestCase):
    def test_determine_platform_explicit(self):
        self.assertEqual(determine_platform("A2"), "A2")
        self.assertEqual(determine_platform("A3"), "A3")
        self.assertEqual(determine_platform("A5"), "A5")

    @patch.dict(sys.modules, {"torch": MagicMock(), "torch.npu": MagicMock()})
    def test_determine_platform_auto_910B(self):
        import torch

        torch.npu.is_available.return_value = True

        mock_props = MagicMock()
        mock_props.name = "Ascend910B"
        torch.npu.get_device_properties.return_value = mock_props

        self.assertEqual(determine_platform("auto"), "A2")

    @patch.dict(sys.modules, {"torch": MagicMock(), "torch.npu": MagicMock()})
    def test_determine_platform_auto_910C(self):
        import torch

        torch.npu.is_available.return_value = True

        mock_props = MagicMock()
        mock_props.name = "Ascend910_93"
        torch.npu.get_device_properties.return_value = mock_props

        self.assertEqual(determine_platform("auto"), "A3")

    @patch.dict(sys.modules, {"torch": MagicMock(), "torch.npu": MagicMock()})
    def test_determine_platform_auto_950D(self):
        import torch

        torch.npu.is_available.return_value = True

        mock_props = MagicMock()
        mock_props.name = "Ascend950"
        torch.npu.get_device_properties.return_value = mock_props

        self.assertEqual(determine_platform("auto"), "A5")

    @patch.dict(sys.modules, {"torch": MagicMock(), "torch.npu": MagicMock()})
    def test_determine_platform_auto_fallback(self):
        import torch

        torch.npu.is_available.return_value = False

        # When NPU is not available, it should fallback to A3
        self.assertEqual(determine_platform("auto"), "A3")


if __name__ == "__main__":
    unittest.main()
