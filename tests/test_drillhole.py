import unittest
from src.drillhole import Drillhole

import pandas as pd
import numpy as np


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

        self.desurvey_method: list[str] = [
            "mininum_curvature",
            "radius_curvature",
            "average_tangent",
            "balanced_tangent",
            "high_tangent",
            "low_tangent",
        ]
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

    def test_DesurveyMethodWithSurvey(self):
        '''
        test desurveying when there is a survey        
        '''

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
        '''
        test desurveying when there is no survey        
        '''

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
        '''
        test desurveying when the survey table is a single item        
        '''

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
            Drillhole('a',-10,1,1,1,1,1)

    def test_TriggerAziError(self):
        with self.assertRaises(ValueError):
            Drillhole('a',10,-1,10,1,1,1)

    def test_TriggerIncError(self):
        with self.assertRaises(ValueError):
            Drillhole('a',10,1,-10,1,1,1)
    def test_EmptyDF(self):
        Drillhole('a',10,1,10,1,1,1,survey=pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
