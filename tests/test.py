import unittest

from src.drillhole import Drillhole
from src.drillhole import Drillhole, IntervalData,PointData
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
        self.simplify = {"A": {1: 9, 2: -1}, "B": {"a": "A"}}

    def test_Create(self):
        d = IntervalData(self.test)

    def test_IntervalSimplify(self):
        d = IntervalData(self.test, column_map=self.simplify)

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


class TestDrillhole(unittest.TestCase):

    def setUp(self):

        self.assay = pd.DataFrame(
            {"depthfrom": range(10), "depthto": range(1, 11), "Fe": range(10)}
        )
        self.survey = pd.DataFrame(
            {"depth": [4, 9], "inclination": [-90, -85], "azimuth": [0, 0]}
        )
        self.strat = pd.DataFrame({"depthfrom": [0], "depthto": [9], "strat": ["a"]})

    def test_MakeDrillhole(self):
        dh = Drillhole(
            "a",
            10,
            0,
            0,
            0,
            0,
            0,
            0,
            survey=self.survey,
            strat=self.strat,
            assay=self.assay,
        )


if __name__ == "__main__":
    unittest.main()
