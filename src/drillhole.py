import pandas as pd
from pathlib import Path
import numpy as np
import wellpathpy as wp
from dataclasses import dataclass, field
from typing import Union, Literal, Any
from geodata import PointData, IntervalData
import pyvista as pv
from dipconversion import strikedip2vector

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
        assumed to be point data on import then converted to interval data internally when compositing is required.
        requires a depth column

    struct: pd.DataFrame
        structural data
        point data columns need to be dip and dip direction


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
    struct: DFNone = None
    watertable: Union[float, None] = None
    negative_down:bool = False
    display_resolution: float = 0.1
    desurvey_method: Literal[
        "mininum_curvature",
        "radius_curvature",
        "average_tangent",
        "balanced_tangent",
        "high_tangent",
        "low_tangent",
    ] = "mininum_curvature"

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
        # yes subtract the depth
        self.z = self.elevation - depth
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

    def _check_struct(self):
        """
        ensure that the struct dataframe contains depth, dip and dipdirection
        """
        if isinstance(self.struct, PointData):
            # survey columns
            self.struct = PointData(
                self.struct, extra_validation_columns=["dip", "dipdirection"]
            )

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
        # the below line commented out is for when I was plotting as point data
        # rather than cell data
        # intervals = table[outcolumns].apply(np.repeat, axis=0, repeats=2)
        intervals = table[outcolumns]
        return xyz, intervals

    def _point_vtk_generator(self, table):
        """
        creates intervals data to pass to vtk
        """
        fx, fy, fz = self._interp_survey_depth(table["depth"])
        xyz = np.vstack((fx, fy, fz)).T
        # look for all the extra columns and return them
        outcolumns = table.columns[~table.columns.isin(["depth"])]
        intervals = table[outcolumns]
        return xyz, intervals

    def extract_vtk_data(self):
        """
        extracts the vtk information to create vtk data but does not create the vtk structures
        as running vtk creation is very slow
        """
        out = {}  # list to store the vtk tmp
        for i in self.type_map.keys():
            if i != "survey":
                tmp = getattr(self, i)
                if tmp is not None:
                    if isinstance(tmp, IntervalData):
                        xyz, intervals = self._interval_vtk_generator(tmp)
                        dtype = "intervals"
                    elif isinstance(tmp, PointData):
                        xyz, intervals = self._point_vtk_generator(tmp)
                        dtype = "points"
                    out.update(
                        {i: {"datatype": dtype, "points": xyz, "data": intervals}}
                    )
        return out

    def create_vtk(self):
        """
        creates the vtk drillhole datasets for visualisation
        specifically it generates lines that can be displayed correctly in paraview etc.
        """
        out = {}  # list to store the vtk tmp
        for i in self.type_map.keys():
            if i != "survey":
                tmp = getattr(self, i)
                if tmp is not None:
                    if isinstance(tmp, IntervalData):
                        xyz, intervals = self._interval_vtk_generator(tmp)
                        ll = pv.line_segments_from_points(xyz)
                        for c in intervals.columns:
                            ll.cell_data[c] = intervals[c]

                    elif isinstance(tmp, PointData):
                        xyz, intervals = self._point_vtk_generator(tmp)
                        ll = pv.PointSet(xyz)
                        for c in intervals.columns:
                            ll.point_data[c] = intervals[c]
                    out.update({i: ll})
        return out

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
        dip = dip.copy()
        if isinstance(dip, (float, int)):
            if dip == 180:
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
            "struct": PointData,
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
        # if negative down
        if self.negative_down:
            self.inclination = 90 + self.inclination
            if isinstance(self.survey, PointData):
                self.survey.inclination = self._dip_check(90 + self.survey.inclination)
        # check dip and azi
        self._validate_dipazi(self.inclination, self.azimuth)
        if isinstance(self.survey, PointData):
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
        self._check_struct()
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
    survey: pd.DataFrame = None
    assay: pd.DataFrame = None
    geology: pd.DataFrame = None
    strat: pd.DataFrame = None
    lith: pd.DataFrame = None
    geophysics: pd.DataFrame = None
    watertable: pd.DataFrame = None
    struct: pd.DataFrame = None
    stratcolumn: pd.DataFrame = None
    desurvey_method: Literal[
        "mininum_curvature",
        "radius_curvature",
        "average_tangent",
        "balanced_tangent",
        "high_tangent",
        "low_tangent",
    ] = "mininum_curvature"
    negative_down: bool = True

    def _extract_hole(self, table_name, holeid):
        """
        extracts the individual drill holes from a DataFrame
        """

        tmp = getattr(self, table_name)
        idx = tmp.holeid == holeid
        output = tmp[idx].reset_index(drop=True).copy()
        return output

    def __post_init__(self):
        """
        creates the dataset
        """
        drillholes: list[Drillhole] = []
        # some pre-calculations for speed
        # check for empty/non-existant tables
        full_tables: list = []
        table_names = [
            "survey",
            "assay",
            "geology",
            "strat",
            "geophysics",
            "watertable",
            "struct",
        ]
        for i in table_names:
            if isinstance(getattr(self, i), pd.DataFrame):
                full_tables.append(i)
        # for tables that have information we get the unique bhid's
        # so that we can only execute the indexing if we need to
        table_bhids: dict[str, set] = {}
        for f in full_tables:
            tmp = getattr(self, f)
            table_bhids.update({f: set(tmp.holeid.unique().tolist())})

        ubhid = self.collar.holeid.unique()
        for bhid in tqdm(ubhid):
            tmp_tables = {}
            for t in full_tables:
                if bhid in table_bhids[t]:
                    tmp_data = self._extract_hole(t, bhid)
                else:
                    tmp_data = None
                tmp_tables.update({t: tmp_data})

            cidx = self.collar.holeid == bhid
            tmp_collar = self.collar[cidx]
            tmp = Drillhole(
                bhid,
                tmp_collar.depth.item(),
                tmp_collar.inclination.item(),
                tmp_collar.azimuth.item(),
                tmp_collar.easting.item(),
                tmp_collar.northing.item(),
                tmp_collar.elevation,
                **tmp_tables,
                negative_down=self.negative_down,
            )
            tmp._contacts
            drillholes.append(tmp)
        self.drillholes = drillholes

    def to_vtk(self):
        """
        creates a vtk multiblock dataset
        """
        tables = ["assay", "geology", "strat", "geophysics", "watertable", "struct"]

        bins = {v: {} for v in tables}
        for dh in self.drillholes:
            # loop over each drillhole and create the vtk datasets
            vtk_dict = dh.create_vtk()
            for v in vtk_dict:
                bins[v].update({dh.bhid: vtk_dict[v]})
        multiblocks = {}
        for i in tables:
            tmpbins = bins[i]
            if len(tmpbins) > 0:
                mb = pv.MultiBlock(tmpbins)
                mb = mb.as_polydata_blocks()
                for n, name in enumerate(tmpbins.keys()):
                    mb.set_block_name(n, name)
                multiblocks.update({i: mb})
        return multiblocks

    def to_csv():
        """
        creates a single .csv file with composited intervals
        """
        pass

    def to_loop(
        self, fractional_depth=True, estimate_stratcolumn=True, use_graph_estimate=True
    ):
        """
        generates loop data
        """
        tmp_contacts = []
        for dh in self.drillholes:
            if isinstance(dh.strat, IntervalData):
                tmp = dh.contacts.copy()
                tmp["bhid"] = dh.bhid
                tmp_contacts.append(tmp)

        contacts = pd.concat(tmp_contacts)
        # estimate the stratigraphic column from stratigraphic transitions
        if estimate_stratcolumn:
            import networkx as nx

            cidx = contacts["type"] == "inside"

            units = contacts[cidx]["feature_name"].unique()
            unit_map = {u[1]: u[0] for u in enumerate(units) if not (u[1] == None)}
            back_map = {u[0]: u[1] for u in enumerate(units) if not (u[1] == None)}
            # attempt to estimate the stratigraphic sequence
            # this is sort of like a transition matrix
            nunits = len(unit_map)
            strat_mat = np.zeros([nunits] * 2)
            t = tmp_contacts[0]
            for t in tmp_contacts:
                tidx = t["type"] == "inside"
                if (~t[tidx].feature_name.isna()).all():
                    tmp_seq = t[tidx].feature_name.map(unit_map).values
                    for i in range(len(tmp_seq) - 1):
                        strat_mat[tmp_seq[i], tmp_seq[i + 1]] += 1

            # work out the stratigraphic order
            norm_prob = np.log(strat_mat / np.sum(strat_mat))
            norm_prob[~np.isfinite(norm_prob)] = 0
            g = nx.DiGraph(norm_prob)

            # find nodes with no-incoming connections
            # these are the tops
            tops = []
            for n in range(nunits):
                if g.in_degree(n) == 0:
                    tops.append(n)
            # find nodes with no-outgoing connections
            # these are the bottoms
            bottoms = []
            for n in range(nunits):
                if g.out_degree(n) == 0:
                    bottoms.append(n)
            # nodes with a single out-degrees
            # conformable units
            # these are the bottoms
            bottoms = []
            for n in range(nunits):
                if g.out_degree(n) == 0:
                    bottoms.append(n)

            # find the path that visits all the nodes this is the strat column
            strat_sequence = []
            for paths in nx.all_simple_paths(g, tops[0], bottoms[0]):
                if len(paths) == nunits:
                    strat_sequence = paths
                    break
            strat_column = {back_map[s[1]]: s[0] for s in enumerate(strat_sequence)}

        # this calculation does a fractional depth calculation best done for flat lying orebodies
        # which really is a best guess at getting the stratigraphic order
        idxfrac = contacts.type == "inside"
        max_depth = contacts[idxfrac].groupby("feature_name")["val"].max()
        frac_depth = contacts.val / contacts.feature_name.map(max_depth.to_dict())
        strat_order = (
            contacts[idxfrac]
            .groupby("feature_name")["Z"]
            .max()
            .reset_index()
            .sort_values("Z", ascending=False)
            .reset_index(drop=True)
        )
        # other method of estimating the strat column
        if use_graph_estimate == False:
            strat_column = {}
            for n, i in strat_order.iterrows():
                strat_column.update({i.feature_name: n})
        # subtract not add the fractional depth this should give the right answer with respect to interfaces vs internal points
        contacts.loc[idxfrac, "val"] = (
            contacts[idxfrac].feature_name.map(strat_column) - frac_depth[idxfrac]
        )
        tmdict = {}
        for i in range(int(contacts.val.min()), int(contacts.val.max()) + 1):
            tmdict.update({i: {"min": i, "max": i + 1, "id": i}})

        return contacts, tmdict, strat_column

    def to_omf():
        """
        generates loop data
        """
        pass

    def to_gempy():
        """
        generates gempy data
        """
        pass
