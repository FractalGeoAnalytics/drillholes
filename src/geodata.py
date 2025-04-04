import pandas as pd
import numpy as np
from typing import Union, Literal, Any

DFNone = Union[pd.DataFrame, None]


class PointData(pd.DataFrame):
    """
    PointData subclasses a DataFrame and

    Parameters
    ----------


    """

    def __init__(self, *args, **kwargs):

        # surely there is a better way to add additional kwargs
        if "extra_validation_columns" in kwargs:
            extra_validation_columns = kwargs.pop("extra_validation_columns")
        else:
            extra_validation_columns = None
        super().__init__(*args, **kwargs)
        validation_columns = ["depth"]

        if extra_validation_columns is not None:
            if isinstance(extra_validation_columns, list):
                validation_columns.extend(extra_validation_columns)
            else:
                raise TypeError("extra_validation_columns must be a list")
        column_set = self.columns.isin(validation_columns)
        if sum(column_set) != len(validation_columns):
            column_names = ",".join(self.columns.to_list())
            valnames = ",".join(validation_columns)
            err = "point data must contain columns {} these are the columns provided {}".format(
                valnames, column_names
            )
            raise ValueError(err)

    def to_interval(self, width: Union[None, float] = None):
        """
        converts point data to interval data.
        it is assumed that point data intervals have from and to depths
        halfway between the sequential points
        to do this we need to sort by the depth column to ensure that we calculate the
        correct interval widths
        the option for user specified widths is provided because sometimes the measurement
        intervals are not actually continuous.

        Parameters
        ------------
        width: float
            optional parameter that fixes the width of the intervals
            no checks are done as it is assumed that you know what the number should be
            i.e. you can produce overlapping or underlapping intervals
        Returns
        ------------
        IntervalData
            adds a depthfrom and depthto column to the dataframe all other columns are returned as is

        """
        tmp = self.copy()
        if width == None:
            # calculate the difference in the depths
            depth_difference: pd.Series = self.depth.diff().copy()
            # assume that the first interval has the same length as the second
            depth_difference.loc[0] = depth_difference.values[1]
            # take half of the depth difference
            half_depth: pd.Series = depth_difference / 2
            tmp.insert(0, "depthfrom", self.depth - half_depth)
            tmp.insert(1, "depthto", self.depth + half_depth)
        else:
            tmp.insert(0, "depthfrom", self.depth - width / 2)
            tmp.insert(1, "depthto", self.depth + width / 2)

        # drop the depth column
        tmp.drop(columns=["depth"], inplace=True)
        return IntervalData(tmp)


class IntervalData(pd.DataFrame):
    """
    interval data requires a from and to depth wraps pd.DataFrame but includes
    some simple data validation steps around required column names

    Parameters
    ------------
    bhid: str
        name of the drillhole

    depth: float
    """

    _metadata = ["midpoint"]

    def __arg_popper(self, kewargs, name, default):
        """
        pops out the arguments from kwargs
        """
        # surely there is a better way to add additional kwargs
        if name in kewargs:
            outarg = kewargs.pop(name)
        else:
            outarg = default

        return outarg

    def __init__(self, *args, **kwargs):

        # surely there is a better way to add additional kwargs
        extra_validation_columns = self.__arg_popper(
            kwargs, "extra_validation_columns", None
        )
        column_map = self.__arg_popper(kwargs, "column_map", None)

        super().__init__(*args, **kwargs)
        # check if we have columns called depthfrom, depthto and anything else that we ask for
        validation_columns = ["depthfrom", "depthto"]

        if extra_validation_columns is not None:
            if isinstance(extra_validation_columns, list):
                validation_columns.extend(extra_validation_columns)
            else:
                raise TypeError("extra_validation_columns must be a list")
        column_set = self.columns.isin(validation_columns)
        if sum(column_set) != len(validation_columns):
            column_names = ",".join(self.columns.to_list())
            valnames = ",".join(validation_columns)
            err = "interval data must contain columns {} these are the columns provided {}".format(
                valnames, column_names
            )
            raise ValueError(err)
        # check that the depth to is > than the from
        idx_start_greater_than_end = self.depthfrom >= self.depthto
        if any(idx_start_greater_than_end):
            raise ValueError("Some intervals have lengths <= 0")
        # simplify the columns i.e. rename geologging or stratigraphy to larger groups
        # we need to call this before the compositing of consecutive samples otherwise
        # we end up having multiple repeat samples
        if column_map is not None:
            self = self._simplify(column_map)

    def composite_consecutive(self, column):
        """
        Composites intervals that have the same value into a
        single longer intervals

        Parameters
        ------------
        column:str
            str of the column that you are going to check for repeat intervals
        Returns
        ------------
            IntervalData
            a subclass of pd.DataFrame but with potentially some of the intervals combined
        """
        if self.shape[0] > 1:
            cond = (self[column] != self[column].shift()).cumsum()
            tmp_agg = (
                self.groupby(cond)
                .agg(
                    fr=("depthfrom", "min"),
                    to=("depthto", "max"),
                    strat=(column, "first"),
                )
                .copy()
            )
            tmp_agg.rename(columns={"fr": "depthfrom", "to": "depthto"}, inplace=True)
            tmp_agg.reset_index(drop=True, inplace=True)
            tmp_agg = IntervalData(tmp_agg)
        return tmp_agg

    def _simplify(self, column_map):
        """
        simplifies the columns provided using a dict of dicts
        dict[str:dict[str, str]]

        effectively just wraps pd.DataFrame().replace

        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html

        """

        self.replace(to_replace=column_map, inplace=True)

        return self

    @property
    def midpoint(self):
        """
        returns the mid points of the intervals
        """
        mid_point: pd.Series = (
            self["depthfrom"] + (self["depthto"] - self["depthfrom"]) / 2
        )
        return mid_point

    def to_pointdata(self):
        """
        converts to point data
        """
        tmp = self
        tmp.insert(0, "depth", tmp.midpoint)
        tmp.drop(columns=["depthfrom", "depthto"], inplace=True)
        return PointData(tmp)
