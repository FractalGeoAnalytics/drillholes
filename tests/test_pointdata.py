import unittest

from geodata import PointData

import pandas as pd
import numpy as np

class TestPointData(unittest.TestCase):

    def setUp(self):
        self.test = pd.DataFrame(
            {
                "depth": [0, 1, 2, 3, 4],
                "A": [1, 1, 2, 3, 1],
                "B": ["a", "b", "c", "d", "e"],
            }
        )

    def test_Create(self):
        d = PointData(self.test)

    def test_NameFailureDepth(self):
        "check that we can find the wrong columns"
        tmp = self.test.copy()
        tmp.rename(columns={"depth": "worgn"}, inplace=True)
        with self.assertRaises(ValueError):
            PointData(tmp)

    def test_NameExtraColumns(self):
        tmp = self.test.copy()
        PointData(tmp, extra_validation_columns=["A"])

    def test_NameExtraColumnsFail(self):
        tmp = self.test.copy()
        with self.assertRaises(ValueError):
            PointData(tmp, extra_validation_columns=["Z"])

    def test_NameExtraColumnsFailColumns(self):
        tmp = self.test.copy()
        with self.assertRaises(TypeError):
            PointData(tmp, extra_validation_columns="Z")

    def testConvertToInterval(self):
        tmp = self.test.copy()
        depthto = PointData(tmp).to_interval()["depthto"]
        np.testing.assert_allclose(depthto, [0.5, 1.5, 2.5, 3.5, 4.5])

    def testConvertToIntervalFixedWidth(self):
        tmp = self.test.copy()
        depthto = PointData(tmp).to_interval(width=1)["depthto"]
        np.testing.assert_allclose(depthto, [0.5, 1.5, 2.5, 3.5, 4.5])


if __name__ == "__main__":
    unittest.main()
