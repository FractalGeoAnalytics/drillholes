import unittest

from geodata import IntervalData

import pandas as pd
import numpy as np


class TestIntervalData(unittest.TestCase):

    def setUp(self):
        self.test = pd.DataFrame(
            {
                "depthfrom": [0, 1, 2, 3, 4],
                "depthto": [1, 2, 3, 4, 5],
                "A": [1, 1, 2, 3, 1],
                "B": ["a", "b", "c", "d", "e"],
            }
        )
        self.fail_length = pd.DataFrame(
            {
                "depthfrom": [0, 1, 2, 3, 4],
                "depthto": [0, 2, 3, 4, 5],
                "A": [1, 1, 2, 3, 1],
                "B": ["a", "b", "c", "d", "e"],
            }
        )
        self.simplify = {"A": {1: 9, 2: -1}, "B": {"a": "A"}}

    def test_Create(self):
        d = IntervalData(self.test)

    def test_IntervalSimplify(self):
        d = IntervalData(self.test, column_map=self.simplify)

    def test_CompositeConsecutive(self):
        d = IntervalData(self.test).composite_consecutive('A')

    def test_NameFailureFrom(self):
        "check that we can find the wrong columns"
        tmp = self.test.copy()
        tmp.rename(columns={"depthfrom": "worgn"}, inplace=True)
        with self.assertRaises(ValueError):
            IntervalData(tmp, column_map=self.simplify)

    def test_NameFailureTo(self):
        tmp = self.test.copy()
        tmp.rename(columns={"depthto": "worgn"}, inplace=True)
        with self.assertRaises(ValueError):
            IntervalData(tmp, column_map=self.simplify)

    def test_NameFailureBoth(self):
        tmp = self.test.copy()
        tmp.rename(columns={"depthto": "worgn", "depthfrom": "asvs"}, inplace=True)
        with self.assertRaises(ValueError):
            IntervalData(tmp, column_map=self.simplify)

    def test_NameExtraColumns(self):
        tmp = self.test.copy()
        IntervalData(tmp, extra_validation_columns=["A"])

    def test_NameExtraColumnsFail(self):
        tmp = self.test.copy()
        with self.assertRaises(ValueError):
            IntervalData(tmp, extra_validation_columns=["Z"])

    def test_NameExtraColumnsNotList(self):
        tmp = self.test.copy()
        with self.assertRaises(TypeError):
            IntervalData(tmp, extra_validation_columns="Z")

    def test_FailLength(self):
        tmp = self.fail_length.copy()
        with self.assertRaises(ValueError):
            IntervalData(tmp)

    def test_MidPoint(self):
        tmp = self.test.copy()
        np.testing.assert_allclose(
            IntervalData(tmp).midpoint, [0.5, 1.5, 2.5, 3.5, 4.5]
        )
    def test_ToIntervalData(self):
        tmp = self.test.copy()
        tmp2 = IntervalData(tmp).to_pointdata()
        np.testing.assert_allclose(tmp2['depth'], [0.5, 1.5, 2.5, 3.5, 4.5])


if __name__ == "__main__":
    unittest.main()
