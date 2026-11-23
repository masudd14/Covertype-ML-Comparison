"""Unit tests for Covertype ML comparison utilities."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN, ALL_COLUMNS, COVER_TYPE_LABELS
from src.evaluate import get_performance_summary


class TestCovertypePipeline(unittest.TestCase):
    """Test suite for pipeline components."""

    def test_column_definitions(self):
        """Ensure column schema and class counts match UCI specifications."""
        self.assertEqual(len(FEATURE_COLUMNS), 54)
        self.assertEqual(TARGET_COLUMN, "Cover_Type")
        self.assertEqual(len(ALL_COLUMNS), 55)
        self.assertEqual(len(COVER_TYPE_LABELS), 7)

    def test_performance_summary(self):
        """Verify performance metrics calculation on synthetic predictions."""
        y_true = np.array([1, 2, 3, 4, 5, 6, 7])
        y_pred = np.array([1, 2, 3, 4, 5, 6, 7])
        res = get_performance_summary(y_true, y_pred)

        self.assertAlmostEqual(res["accuracy"], 1.0)
        self.assertAlmostEqual(res["macro_f1"], 1.0)
        self.assertAlmostEqual(res["weighted_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
