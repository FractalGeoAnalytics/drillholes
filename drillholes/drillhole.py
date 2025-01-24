import pandas as pd
from pathlib import Path
import numpy as np
import wellpathpy as wp
from dataclasses import dataclass,field
from typing import Union
from datetime import datetime
DFNone = Union[pd.DataFrame, None]
from typing import Literal

@dataclass
class Drillhole:
    """
    Class to manage drill hole data geological, survey, assay and geophysical data can be stored in a drillhole
    This class offers simple 2d and 3d plotting utilities desurveying and compositing

    """
    bhid: str # name of the drill hole
    depth: float
    inclination: float
    azimuth: float
    easting: float
    northing: float
    elevation: float
    grid: Union[str, None] = None
    drilldate:Union[str, None] = None
    survey: DFNone = None
    assay: DFNone = None
    geology:DFNone = None
    strat: DFNone = None
    geophysics: DFNone = None
    watertable: DFNone = None

    desurvey_method:Literal['mininum_curvature', 'radius_curvature', 'average_tangent', 'balanced_tangent','high_tangent','low_tangent'] = 'mininum_curvature'
    positive_down:bool = True
    def _desurvey(self):
        """
        desurvey the drill hole location using either the
        collar information or the survey if there is
        if a survey exists that is preferred
        """
        tangent_mapper:dict[str]= {'average_tangent':'ave','balanced_tangent':'bal','high_tangent':'high','low_tangent':'low'}
        hasSurvey = False
        # surveys that are 1 row are planned surveys and thus we handle them 
        # by using their values as the survey.
        if isinstance(self.survey, pd.DataFrame):
            if self.survey.shape[0] == 1:
                hasSurvey = False
                self.inclination = self.survey.inclination.item()
                self.azimuth = self.survey.azimuth.item()
            else:
                hasSurvey = True
                # ensure that the survey is sorted by depth
                self.survey = self.survey.sort_values(by='depth')
        if hasSurvey:
            # if there is a survey we need to ensure that the survey depth 
            # is at least as deep as the collar
            if self.survey.depth.max() < self.depth:
                # extend the dip and azi to the collar depth
                last_point = self.survey.iloc[-1:].copy()
                last_point.depth = self.depth
                self.survey = pd.concat([self.survey,last_point])
            desurvey = wp.deviation(self.survey.depth, self.survey.inclination, self.survey.azimuth)
            self.survey_depth = self.survey.depth
        else:
            # make the survey depth
            self.survey_depth = np.asarray([0,self.depth])
            desurvey = wp.deviation(self.survey_depth, [self.inclination]*2, [self.azimuth]*2)
        if self.desurvey_method == 'mininum_curvature':
            tmp = desurvey.minimum_curvature()
        elif self.desurvey_method == 'radius_curvature':
            tmp = desurvey.radius_curvature()
        elif self.desurvey_method in ('average_tangent','balanced_tangent','high_tangent','low_tangent'):
            tmp = desurvey.tan_method(tangent_mapper[self.desurvey_method])
        depth = tmp.depth
        east = tmp.easting
        north = tmp.northing
        # add the easting ,northing and elevation create xyz for each point
        self.x = east+self.easting
        self.y = north+self.northing
        self.z = depth+self.elevation
        if hasSurvey:
            self.survey_type = 'collar'
        else:
            self.survey_type = 'survey'
        return self

    def _check_survey(self):
        '''
        ensure that the survey dataframe contains depth, inclination and azimuth columns
        '''
        if isinstance(self.survey,pd.DataFrame): 
            # survey columns
            column_set = self.survey.columns.isin(['depth','inclination','azimuth'])
            if sum(column_set) != 3:
                column_names = ','.join(self.survey.columns.to_list())
                err = 'survey columns must only have depth, inclination and azimuth these are column names provided {}'.format(column_names)
                raise ValueError(err)
            else:

                self._validate_dipazi(self.survey.inclination, self.survey.azimuth)
                
    def _composite():
        """
        code that composites:
        composites assays to regular intervals of stratigraphy 
        composites geophysics to assay intervals

        """
        pass

    def _check_stratigraphy(self):
        """
        checks the stratigraphy we have two distinct categories holes that are totally inside a unit
        and contacts this function calculates both 
        loop3d interfaces and value constraints

        """

        pass

    def _create_scalar_field(self):
        '''
        assumes that the scalar field is perpendicular to the drillholes and scalar field is 
        equivalent to the depth ideally if there is a downhole dip measurement we calculate the scalar 
        field perpendicular to that
        ====
        NO Actually for drill holes we might need to try calculating a scalar field that
        is relative to 0 being the contact of the field and model in loop with multiple instances
        probably setting 1 to above the contact and -1 to below the contact this assumes that the drillhole is
        perpendicular to dip contact is 0

        In the case where the hole does not contain a contact this is something else maybe just 
        actually we can't use a true depth for a scalar field so we use -1,0,+1 
        then maybe do some trickery to force the column

        Yes this is correct also conformable layers need to be a contact and share x, y,z values 
        [0,0,0,0]
        [0,0,0,1]
        '''

    def _create_contacts(self):
        """
        creates a dataframe suitable for conversion to loop3d/gempy modelling
        """

        if isinstance(self.strat,pd.DataFrame):
            nlith = self.strat.shape[0]
            if nlith == 1:
                # need at least 2 lith to have a contact thus the first 
                mid = self._mid_point(self.strat['from'],self.strat['to']).values
                strat = self.strat['strat'].values
                val = 1
                tmp = pd.DataFrame({'depth':mid,'strat':strat,'val':val,'type':'inside'},index=[0])
                self._inside = tmp
            if nlith >1 :
                mids = self._mid_point(self.strat['from'],self.strat['to']).values
                self._inside = pd.DataFrame({'depth':mids,'strat':self.strat['strat'].values,'type':['inside']*nlith})
                val = mids
                self._inside['val'] = val
                # we are looking for the bottoms but this doesn't include the last interval or the first
                tmp_b = self.strat[['to','strat']][1:-1].copy().reset_index(drop=True)
                tmp_b = tmp_b.rename(columns={'to':'depth'})
                tmp_b['type'] = 'bottom'
                # get the tops of the other units 
                tmp_t = self.strat[['from','strat']][2:].copy().reset_index(drop=True)
                tmp_t = tmp_t.rename(columns={'from':'depth'})
                tmp_t['type'] = 'top'
                conformables = pd.concat([tmp_b,tmp_t]).reset_index(drop=True)
                self._contacts = conformables
        return self

    def _interp_survey_depth(self, depth):
        '''
        interpolates uses the depth of an object to interpolate it's desurveyed position 
        '''
        x:np.ndarray = np.interp(depth,self.survey_depth,self.x)
        y:np.ndarray = np.interp(depth,self.survey_depth,self.y)
        z:np.ndarray = np.interp(depth,self.survey_depth,self.z)
        return x,y, z

    def _desurvey_strat(self):
        '''
        desurveys the stratigraphy to xyz
        '''
        tmp = []
        if hasattr(self,'_contacts'):
            bx,by,bz = self._interp_survey_depth(self._contacts.depth)
            txyz = pd.DataFrame({'X':bx,'Y':by,'Z':bz})
            txyz['feature_name'] = self._contacts.strat
            txyz['type'] = self._contacts['type']

            tmp.append(txyz)
        if hasattr(self,'_inside'):
            bx,by,bz = self._interp_survey_depth(self._inside.depth)
            txyz = pd.DataFrame({'X':bx,'Y':by,'Z':bz})
            txyz['feature_name'] = self._inside.strat
            txyz['type'] = self._inside['type']
            txyz['val'] = self._inside['val']
            tmp.append(txyz)
        if len(tmp)>0:
            out = pd.concat(tmp)
            self.contacts = out
        return self
    
    def _check_overlap(self,x):
        """
        ensures that intervals don't overlap if it does throw a warning
        """
        x.TO-x.FROM
        
        pass

    def _mid_point(self,depthfrom,depthto):
        '''
        calculates the mid point
        '''
        mids = depthfrom+(depthto-depthfrom)/2
        return mids
    def _check_assay(self):
        '''
        ensure that the assay dataframe contains from and to columns
        '''
        if isinstance(self.assay,pd.DataFrame): 
            # survey columns
            column_set = self.assay.columns.isin(['depthfrom','depthto'])
            if sum(column_set) != 2:
                column_names = ','.join(self.assay.columns.to_list())
                err = 'assays must contain columns from and to these are the columns provided {}'.format(column_names)
                raise ValueError(err)
            else:
                pass
    def _desurvey_assays(self):
        '''
        desurveys the assays to xyz
        '''
        if isinstance(self.assay,pd.DataFrame):
            samp_len:np.ndarray =(self.assay.depthto.values-self.assay.depthfrom.values)
            mids:np.ndarray = self.assay.depthfrom.values+samp_len/2
            x:np.ndarray = np.interp(mids,self.survey_depth,self.x)
            y:np.ndarray = np.interp(mids,self.survey_depth,self.y)
            z:np.ndarray = np.interp(mids,self.survey_depth,self.z)
            self.assay_x = x
            self.assay_y = y
            self.assay_z = z
        return self


    def _check_geophysics(self):
        """
        some checks for geophysics assume that 
        """
        pass

    def _dip_check(self,dip):
        '''
        function only to ensure dips are less than 180
        '''
        # special case of the dip starting at 180 exactly meaning that the hole is going straight up
        # we handle this by setting the dip to 179.99999 avoid the errors probably this is acceptable.
        if isinstance(dip,(float, int)):
            dip = self.__magic_dip
        else:
            dip[dip == 180] = self.__magic_dip
        return dip 

    def _validate_dipazi(self, dip, azi):
        """
        ensure that the dip and azimuth are within the expectation of wellpathpy
        """
        dipcheck = (dip >= 0) & (dip < 180)
        azicheck = (azi >= 0) & (azi < 360)

        dip_ok = np.all(dipcheck)
        azi_ok = np.all(azicheck)
        dip_message: str = ""
        if not dip_ok:
            dip_message = "all dips must be >=0 and <180 negative dips are not allowed wellpathpy assumes 0 is a vertical hole.\n"
        azi_message: str = ""
        if not azi_ok:
            azi_message = "all azimuths must be >=0 and <360"
        outmessage = dip_message + azi_message
        if not azi_ok or not dip_ok:
            raise ValueError(outmessage)

    def __convert_pd_to_pythontypes(self,x,name):
        '''converts to native types'''

        if isinstance(x,(float, int, str)):
            lenx = 0
        elif isinstance(x, (pd.Series,pd.DataFrame)):
            lenx = len(x)
        elif x == None:
            lenx = 0
        else:
            lenx = len(x)

        if lenx >1:
            raise ValueError(f'too many items in {name}')
        # if we have a data frame or series of a single row assume that 
        # we have the right data and extract it
        # if there is more than 1 row throw error
        if isinstance(x,pd.core.series.Series):
            y = x.item()
        elif isinstance(x,pd.core.frame.DataFrame):
            y = x.values.item()
        else:
            y = x
        return y

    def __post_init__(self):
        # magic number for vertical holes
        self.__magic_dip = 179.9999999
        # deep copy any dataframes that are used to prevent changes propagating.
        for i in ['survey','assay','geology','strat','geophysics','watertable']:
            tmp = getattr(self, i)
            if isinstance(tmp,pd.DataFrame):
                # good idea to reset index too
                new = tmp.copy(deep=True).reset_index(drop=True)
                setattr(self,i,new)
        # check that the inputs types are acceptable
        python_basetypes = ['bhid','depth','inclination','azimuth','easting','northing','elevation','grid','drilldate']
        for i in python_basetypes:
            tmp = self.__convert_pd_to_pythontypes(getattr(self,i),i)
            setattr(self, i,tmp)
        # check dip and azi
        if self.positive_down:
            self.inclination = 90+self.inclination
            if isinstance(self.survey, pd.DataFrame):
                self.survey.inclination = self._dip_check(90+self.survey.inclination)

        self.inclination = self._dip_check(self.inclination)
        if isinstance(self.survey, pd.DataFrame):
            self.inclination = self._dip_check(self.survey.inclination)

        self._validate_dipazi(self.inclination, self.azimuth)
        self._check_survey()
        self._desurvey()
        self._check_assay()
        self._desurvey_assays()
        self._check_stratigraphy()
        self._create_contacts()
        self._desurvey_strat()
        # check hole depth
        if self.depth <=0:
            raise ValueError('Hole Depth must be >0')
        return self

    def __repr__(self):
        output:str = 'BHID:{}\nDepth:{}\nDIP:{}\nAZI:{}'.format(self.bhid,self.depth,self.inclination, self.azimuth)
        return output

