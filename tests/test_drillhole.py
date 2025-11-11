import unittest
from src.drillhole import Drillhole

import pandas as pd
import numpy as np
from itertools import combinations
from functools import partial


class TestDrillhole(unittest.TestCase):

    def setUp(self):

        self.assay = pd.DataFrame(
            {"depthfrom": range(10), "depthto": range(1, 11), "Fe": range(10)}
        )
        self.survey = pd.DataFrame(
            {"depth": [4, 9], "inclination": [-90, -85], "azimuth": [0, 0]}
        )
        self.strat = pd.DataFrame(
            {"depthfrom": [0, 9], "depthto": [9, 10], "strat": ["a", "b"]}
        )
        self.survey = pd.DataFrame({'depth': {0: 0.0, 1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0, 5: 50.0, 6: 59.0}, 'azimuth': {0: 304.11, 1: 303.01, 2: 302.72, 3: 300.45, 4: 300.6, 5: 299.36, 6: 298.38}, 'inclination': {0: 29.32, 1: 29.25, 2: 29.009999999999998, 3: 28.409999999999997, 4: 27.299999999999997, 5: 26.299999999999997, 6: 90-25.599999999999994}})
        self.geophys = pd.DataFrame({"depth": [0, 1, 2], "geophys": [1, 2, 3]})

        self.geology = pd.DataFrame(
            {"depthfrom": [0, 1, 2], "depthto": [1, 2, 3], "geology": [1, 2, 3]}
        )
        self.watertable = 10
        self.desurvey_method: list[str] = [
            "mininum_curvature",
            "radius_curvature",
            "average_tangent",
            "balanced_tangent",
            "high_tangent",
            "low_tangent",
        ]

    def test_MakeDrillholeDataFrameInputsPermuter(self):

        # combinations and permutations of input data frames
        input_parameters = ["geophys", "assay", "strat", "geology", "watertable"]
        for n in range(1, len(input_parameters) + 1):
            for i in combinations(input_parameters, n):
                test = {j: getattr(self, j) for j in i}
                dh = Drillhole("a", 10, 0, 0, 0, 0, 0, 0, *test)

    def test_MakeDrillholeStratSingular(self):
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
            strat=self.strat.iloc[1:],
            assay=self.assay,
        )

    def test_DesurveyMethodWithSurvey(self):
        """
        test desurveying when there is a survey
        """

        for i in self.desurvey_method:
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
                desurvey_method=i,
            )

    def test_DesurveyMethodWithoutSurvey(self):
        """
        test desurveying when there is no survey
        """

        for i in self.desurvey_method:
            dh = Drillhole(
                "a",
                10,
                0,
                90,
                0,
                0,
                0,
                0,
                strat=self.strat,
                assay=self.assay,
                desurvey_method=i,
            )

    def test_DesurveyMethodWithSurveySingleObs(self):
        """
        test desurveying when the survey table is a single item
        """

        for i in self.desurvey_method:
            dh = Drillhole(
                "a",
                10,
                0,
                90,
                0,
                0,
                0,
                0,
                survey=self.survey.iloc[0:1].copy(),
                strat=self.strat,
                assay=self.assay,
                desurvey_method=i,
            )

    def test_Simplify(self):
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
            strat_simplify={"strat": {"b": "a"}},
        )

    def test_TriggerDepthError(self):
        with self.assertRaises(ValueError):
            Drillhole("a", -10, 1, 1, 1, 1, 1)

    def test_TriggerAziError(self):
        with self.assertRaises(ValueError):
            Drillhole("a", 10, -1, 10, 1, 1, 1, negative_down=False)

    def test_TriggerAziErrorPosDown(self):
        with self.assertRaises(ValueError):
            Drillhole("a", 10, 99, 10, 1, 1, 1, negative_down=True)

    def test_AziOKPosDown(self):
        for i in range(0,91):
            Drillhole("a", 10, i, 10, 1, 1, 1, negative_down=False)

    def test_AziOKPosUp(self):
        Drillhole("a", 10, -90, 10, 1, 1, 1, negative_down=True)

    def test_EmptyDF(self):
        Drillhole("a", 10, 1, 10, 1, 1, 1, survey=pd.DataFrame())
    
    def test_NegFail(self):
        Drillhole("a", 10, 1, 10, 1, 1, 1, survey=self.survey, negative_down=False)
    

if __name__ == "__main__":
    unittest.main()
