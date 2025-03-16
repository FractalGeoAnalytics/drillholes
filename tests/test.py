import unittest

from src.drillhole import Drillhole
from src.drillhole import Drillhole,IntervalData
import pandas as pd
import numpy as np

class TestIntervalData(unittest.TestCase):
    
    def setUp(self):
        self.test = pd.DataFrame({'depthfrom':[0,1,2,3,4],'depthto':[1,2,3,4,5],'A':[1,1,2,3,1],'B':['a','b','c','d','e']})
        self.simplify = {'A':{1:9,2:-1},'B':{'a':'A'}}
    
    def test_IntervalData(self):
        d = IntervalData(self.test, composite_column='A')

    def test_IntervalSimplify(self):
        d = IntervalData(self.test,column_map=self.simplify)

    def test1(self):
        tt = self.test
        tt.replace(to_replace=)

class TestDrillhole(unittest.TestCase):

    def setUp(self):

        self.assay = pd.DataFrame({'depthfrom':range(10),'depthto':range(1,11),'Fe':range(10)})
        self.survey = pd.DataFrame({'depth':[4,9],'inclination':[-90,-85],'azimuth':[0,0]})
        self.strat = pd.DataFrame({'depthfrom':[0],'depthto':[9],'strat':['a']})

    def test_MakeDrillhole(self):
        dh = Drillhole('a',10,0,0,0,0,0,0,survey=self.survey,strat=self.strat,assay=self.assay)



if __name__ == "__main__":
    unittest.main()



