import pandas as pd
from pathlib import Path
import numpy as np
import wellpathpy as wp
from dataclasses import dataclass, field
from typing import Union, Literal, Any
from datetime import datetime
from geodata import PointData, IntervalData
import pyvista as pv

DFNone = Union[pd.DataFrame, None]
from tqdm import tqdm


@dataclass
class Drillhole:
    """
    Class to manage drill hole data, geological, survey, assay and geophysical data can be stored in a drillhole
    This class offers simple 2d and 3d plotting utilities desurveying and compositing

    Parameters
    ------------
    bhid: str
        name of the drillhole

    depth: float
        end of hole depth of the drill hole

    inclination: float
        dip of the drill hole

    azimuth: float
        direction of the drillhole

    easting: float
        x location

    northing: float
        y location

    elevation: float
        z location

    drilldate: str
        date of drilling

    grid: str
        crs information

    survey: pd.DataFrame
        downhole survey information this overrides the inclination and azimuth
        information when desurveying columns must be depth, inclination and azimuth

    assay: pd.DataFrame
        assay data must contain columns from and to all other columns are assumed to be floats
        i.e. assays and that you can take a weighted average of their results

    geology: pd.DataFrame
        geology must contain columns from and to geology is assumed to be so recompositing will be majority coded or
        fractional geology differs from stratigraphy in we assume here that geology is field logging as opposed to interpretion

    strat: pd.DataFrame
        strat must contain columns from and to geology is assumed to be so recompositing will be majority coded or
        fractional strat differs from geology as we assume that stratigraphy is interpreted later on

    geophysics: pd.DataFrame
        geophysics (wireline logging)

    """

    bhid: str  # name of the drill hole
    depth: float
    inclination: float
    azimuth: float
    easting: float
    northing: float
    elevation: float
    grid: Union[str, None] = None
    drilldate: Union[str, None] = None
    survey: DFNone = None
    assay: DFNone = None
    geology: DFNone = None
    strat: DFNone = None
    strat_simplify: Union[dict[str, dict[str, Any]], None] = None
    geophysics: DFNone = None
    watertable: Union[float, None] = None
    display_resolution: float = 0.1
    desurvey_method: Literal[
        "mininum_curvature",
        "radius_curvature",
        "average_tangent",
        "balanced_tangent",
        "high_tangent",
        "low_tangent",
    ] = "mininum_curvature"
    positive_down: bool = True

    def _desurvey(self):
        """
        desurvey the drill hole location using either the
        collar information or the survey if there is
        if a survey exists that is preferred
        """
        tangent_mapper: dict[str] = {
            "average_tangent": "avg",
            "balanced_tangent": "bal",
            "high_tangent": "high",
            "low_tangent": "low",
        }
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
                self.survey = self.survey.sort_values(by="depth")
        if hasSurvey:
            # if there is a survey we need to ensure that the survey depth
            # is at least as deep as the collar
            if self.survey.depth.max() < self.depth:
                # extend the dip and azi to the collar depth
                last_point = self.survey.iloc[-1:].copy()
                last_point.depth = self.depth
                self.survey = pd.concat([self.survey, last_point])
            desurvey = wp.deviation(
                self.survey.depth, self.survey.inclination, self.survey.azimuth
            )
            self.survey_depth = self.survey.depth
        else:
            # make the survey depth
            self.survey_depth = np.asarray([0, self.depth])
            desurvey = wp.deviation(
                self.survey_depth, [self.inclination] * 2, [self.azimuth] * 2
            )
        tangent_options: tuple[str] = (
            "average_tangent",
            "balanced_tangent",
            "high_tangent",
            "low_tangent",
        )
        if self.desurvey_method == "mininum_curvature":
            tmp = desurvey.minimum_curvature()
        elif self.desurvey_method == "radius_curvature":
            tmp = desurvey.radius_curvature()
        elif self.desurvey_method in tangent_options:
            tmp = desurvey.tan_method(tangent_mapper[self.desurvey_method])
        depth = tmp.depth
        east = tmp.easting
        north = tmp.northing
        # add the easting ,northing and elevation create xyz for each point
        self.x = east + self.easting
        self.y = north + self.northing
        self.z = depth + self.elevation
        if hasSurvey:
            self.survey_type = "collar"
        else:
            self.survey_type = "survey"
        return self

    def _check_survey(self):
        """
        ensure that the survey dataframe contains depth, inclination and azimuth columns
        """
        if isinstance(self.survey, PointData):
            # survey columns
            self.survey = PointData(
                self.survey, extra_validation_columns=["inclination", "azimuth"]
            )

            self._validate_dipazi(self.survey.inclination, self.survey.azimuth)

    def _check_stratigraphy(self):
        """
        runs simple checks on the quality of the log
        at this moment aggreagates consecutive intervals with the same
        code into a single interval

        """
        if isinstance(self.strat, IntervalData):
            if self.strat.shape[0] > 1:
                tmpstrat = IntervalData(
                    self.strat, extra_validation_columns=["strat"]
                ).composite_consecutive(column="strat")
                self.strat = tmpstrat
        return self

    def _create_scalar_field(self):
        """
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
        """

    def _calculate_inside(self):
        val = self.strat.midpoint
        strat = self.strat["strat"].values
        n_items = len(val)
        if n_items > 1:
            index = np.arange(n_items)
        else:
            index = [0]

        tmp = PointData(
            {"depth": val, "strat": strat, "val": val, "type": ["inside"] * n_items},
            index=index,
        )
        # include the from and to to estimate thickness
        tmp["depthfrom"] = self.strat["depthfrom"].values
        tmp["depthto"] = self.strat["depthto"].values
        self._inside = tmp
        return self

    def _create_contacts(self):
        """
        creates a dataframe suitable for conversion to loop3d/gempy modelling
        extracts the formation tops for geological modelling
        """

        if isinstance(self.strat, IntervalData):
            self = self._calculate_inside()
            # we are looking for the bottoms but this doesn't include the last interval or the first
            tmp_b = self.strat[["depthto", "strat"]][:-1].copy().reset_index(drop=True)
            tmp_b = tmp_b.rename(columns={"depthto": "depth"})
            tmp_b["type"] = "bottom"
            # get the tops of the other units
            tmp_t = self.strat[["depthfrom", "strat"]][1:].copy().reset_index(drop=True)
            tmp_t = tmp_t.rename(columns={"depthfrom": "depth"})
            tmp_t["type"] = "top"
            # copy the bottom and create a new table tmp_c aka contacts
            tmp_c = tmp_b.copy()

            tmp_c["strat"] = (
                tmp_b["strat"].astype(str) + "-" + tmp_t["strat"].astype(str)
            )
            tmp_c["type"] = "contact"
            conformables = pd.concat([tmp_b, tmp_t, tmp_c]).reset_index(drop=True)
            self._contacts = conformables
        return self

    def _interp_survey_depth(self, depth):
        """
        uses the depth of an object to interpolate it's desurveyed position
        """
        x: np.ndarray = np.interp(depth, self.survey_depth, self.x)
        y: np.ndarray = np.interp(depth, self.survey_depth, self.y)
        z: np.ndarray = np.interp(depth, self.survey_depth, self.z)
        return x, y, z

    def _interval_vtk_generator(self, table):
        """
        creates intervals data to pass to vtk
        """
        fx, fy, fz = self._interp_survey_depth(table["depthfrom"])
        tx, ty, tz = self._interp_survey_depth(table["depthto"])
        f = np.vstack((fx, fy, fz)).T
        t = np.vstack((tx, ty, tz)).T
        xyz = np.hstack([f, t]).reshape(-1, 3)
        # look for all the extra columns and return them
        outcolumns = table.columns[~table.columns.isin(["depthfrom", "depthto"])]
        intervals = table[outcolumns].apply(np.repeat, axis=0, repeats=2)

        return xyz, intervals

    def create_vtk(self):
        """
        creates the vtk drillhole datasets for visualisation
        specifically it generates lines that can be displayed correctly in paraview etc.
        """
        for i in self.type_map.keys():
            if i != 'survey':
                tmp = getattr(self, i)
                if tmp is not None:
                    xyz, intervals = self._interval_vtk_generator(tmp)

        return xyz, intervals

    def _desurvey_strat(self):
        """
        desurveys the stratigraphy to xyz
        """
        tmp = []
        if hasattr(self, "_contacts"):
            bx, by, bz = self._interp_survey_depth(self._contacts.depth)
            txyz = pd.DataFrame({"X": bx, "Y": by, "Z": bz})
            txyz["feature_name"] = self._contacts.strat
            txyz["type"] = self._contacts["type"]

            tmp.append(txyz)
        if hasattr(self, "_inside"):
            bx, by, bz = self._interp_survey_depth(self._inside.depth)
            txyz = pd.DataFrame({"X": bx, "Y": by, "Z": bz})
            txyz["feature_name"] = self._inside.strat
            txyz["type"] = self._inside["type"]
            txyz["val"] = self._inside["val"]
            txyz["val"] = self._inside["val"]

            tmp.append(txyz)
        if len(tmp) > 0:
            out = pd.concat(tmp)
            self.contacts = out
        return self

    def _check_assay(self):
        """
        ensure that the assay dataframe contains from and to columns
        """
        if isinstance(self.assay, IntervalData):
            self.assay = IntervalData(self.assay)

    def _desurvey_assays(self):
        """
        desurveys the assays to xyz
        """
        if isinstance(self.assay, IntervalData):
            mids: np.ndarray = self.assay.midpoint
            x: np.ndarray = np.interp(mids, self.survey_depth, self.x)
            y: np.ndarray = np.interp(mids, self.survey_depth, self.y)
            z: np.ndarray = np.interp(mids, self.survey_depth, self.z)
            self.assay_x = x
            self.assay_y = y
            self.assay_z = z
        return self

    def _check_geophysics(self):
        """
        only converts from point to interval data
        """
        self.geophysics = self.geophysics.to_interval()

    def _dip_check(self, dip):
        """
        function only to ensure dips are less than 180
        """
        # special case of the dip starting at 180 exactly meaning that the hole is going straight up
        # we handle this by setting the dip to 179.99999 avoid the errors probably this is acceptable.
        # the other probably better way is to have negative depths so that a vertical hole goes up
        # but that has a lot of downstream issues as you have to flip a heap of parameter.
        if isinstance(dip, (float, int)):
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

    def __convert_pd_to_pythontypes(self, x, name):
        """converts to native types"""

        if isinstance(x, (float, int, str)):
            lenx = 0
        elif isinstance(x, (pd.Series, pd.DataFrame)):
            lenx = len(x)
        elif x == None:
            lenx = 0
        else:
            lenx = len(x)

        if lenx > 1:
            raise ValueError(f"too many items in {name}")
        # if we have a data frame or series of a single row assume that
        # we have the right data and extract it
        # if there is more than 1 row throw error
        if isinstance(x, pd.core.series.Series):
            y = x.item()
        elif isinstance(x, pd.core.frame.DataFrame):
            y = x.values.item()
        else:
            y = x
        return y

    def __post_init__(self):
        # magic number for vertical holes
        self.__magic_dip = 179.9999999
        # deep copy any dataframes that are used to prevent changes propagating.
        # need to do data type conversion here to simplify
        # the processing later on
        type_map = {
            "survey": PointData,
            "assay": IntervalData,
            "geology": IntervalData,
            "strat": IntervalData,
            "geophysics": PointData,
            "watertable": PointData,
        }
        self.type_map = type_map
        for i in type_map:
            tmp = getattr(self, i)
            if isinstance(tmp, pd.DataFrame):
                # good idea to reset index too
                new = tmp.copy(deep=True).reset_index(drop=True)
                # check if the df is empty if so convert to None
                if new.empty:
                    setattr(self, i, None)
                else:
                    setattr(self, i, type_map[i](new))
        # check that the inputs types are acceptable
        python_basetypes: list[str] = [
            "bhid",
            "depth",
            "inclination",
            "azimuth",
            "easting",
            "northing",
            "elevation",
            "grid",
            "drilldate",
        ]
        for i in python_basetypes:
            tmp = self.__convert_pd_to_pythontypes(getattr(self, i), i)
            setattr(self, i, tmp)
        if self.positive_down:
            self.inclination = 90 + self.inclination
            if isinstance(self.survey, pd.DataFrame):
                self.survey.inclination = self._dip_check(90 + self.survey.inclination)
        # check dip and azi
        self._validate_dipazi(self.inclination, self.azimuth)

        self.inclination = self._dip_check(self.inclination)
        if isinstance(self.survey, pd.DataFrame):
            self.inclination = self._dip_check(self.survey.inclination)
        # check hole depth
        if self.depth <= 0:
            raise ValueError("Hole Depth must be >0")

        self._check_survey()
        self._desurvey()
        self._check_assay()
        self._desurvey_assays()
        self._check_stratigraphy()
        self._create_contacts()
        self._desurvey_strat()

        return self

    def __repr__(self):
        output: str = "BHID:{}\nDepth:{}\nDIP:{}\nAZI:{}".format(
            self.bhid, self.depth, self.inclination, self.azimuth
        )
        return output


@dataclass
class DrillData:
    """
    wraps the dataset creation loops so that you only need to pass the
    collar, survey and stratigraphy dataframes

    """

    collar: pd.DataFrame
    survey: pd.DataFrame
    assay: pd.DataFrame
    strat: pd.DataFrame
    lith: pd.DataFrame
    stratcolumn = None
    wireline: pd.DataFrame
    desurvey_method: Literal[   
        "mininum_curvature",
        "radius_curvature",
        "average_tangent",
        "balanced_tangent",
        "high_tangent",
        "low_tangent",
    ] = "mininum_curvature"
    positive_down: bool = True

    def __post_init__(self):
        drillholes: list[Drillhole] = []

        for i in tqdm(self.collars.HOLEID):
            cidx = self.collars.HOLEID == i
            sidx = self.surveys.HOLEID == i
            aidx = self.assay.HOLEID == i
            lidx = self.lithology.holeid == i
            gidx = self.geology.holeid == i
            tmp_survey = self.surveys.loc[sidx, ["DEPTH", "DIP", "AZIMUTH"]].copy()
            tmp_survey.columns = ["depth", "inclination", "azimuth"]

            tmp_collar = self.collars[cidx].copy()
            tmp_assay = self.assay[aidx].copy()
            tmp_lith = self.lithology[lidx]
            tmp_geol = self.geology[gidx]
            tmp = Drillhole(
                i,
                tmp_collar.DEPTH.item(),
                0,
                0,
                tmp_collar.EAST.item(),
                tmp_collar.NORTH.item(),
                tmp_collar.RL,
                survey=tmp_survey,
                assay=tmp_assay,
                strat=tmp_lith,
                geology=tmp_geol,
            )
            drillholes.append(tmp)
        self.drillholes = drillholes

    def to_vtk():
        """
        creates a vtk multiblock dataset
        """
        pass

    def to_csv():
        """
        creates a single .csv file with composited intervals
        """
        pass

    def to_loop():
        """
        loop data
        """
        pass

    def to_gempy():
        """
        gempy data
        """
        pass
