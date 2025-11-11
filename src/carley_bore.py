import pyvista as pv
from LoopStructural import GeologicalModel
from LoopStructural.datatypes import BoundingBox
from LoopStructural.visualisation import Loop2DView, Loop3DView

from matplotlib import pyplot as plt
from pathlib import Path
import shutil
import numpy as np
from augeosciencedatasets import downloaders, readers
import pandas as pd
from matplotlib import pyplot as plt
from drillhole import Drillhole, DrillData
import drillhole
from importlib import reload
reload(drillhole)
outpath = Path("data/ASDCarleyBore")
dasc_urls = {
    "A110793": (
        "Spectra.zip",
        "https://geodocsget.dmirs.wa.gov.au/api/GeoDocsGet?filekey=0586c4d5-1652-48fe-ad0a-a9fec71d7446-834jllo21jrjnkvgrbyqgce5w0sgr16gfot5nfps",
    ),
    "A97345": (
        "Drill.zip",
        "https://geodocsget.dmirs.wa.gov.au/api/GeoDocsGet?filekey=45ba37d4-0c14-440f-b68d-55c273c81721-7q7t19fnlrjaams5rtjl2cifylfkvicgcc9u4w5s",
    ),
    "A108383": (
        "Drill.zip",
        "https://geodocsget.dmirs.wa.gov.au/api/GeoDocsGet?filekey=375b639f-bdfe-402d-bdcc-6792f25588dc-vu7pjtg9j3ym9b4gb16rvcqtrsge9p1ilhxk4dtp",
    ),
}

if not outpath.exists():
    outpath.mkdir(parents=True)

for file in dasc_urls:
    outfile = outpath.joinpath(file, dasc_urls[file][0])
    if not outfile.parent.exists():
        outfile.parent.mkdir(parents=True)
    downloaders.from_dasc(dasc_urls[file][1], outfile)
    # unzip the zip file
    shutil.unpack_archive(outfile, outfile.parent)


tmp_assay = []
tmp_lith = []
tmp_scint = []
tmp_collar = []
tmp_xrf = []
# the column map is to correct some inconsistencies with the column names
column_map = {
    "mFrom": "DEPTH_FROM",
    "mTo": "DEPTH_TO",
    "SITE ID": "Hole_id",
    "SCINTILLOMETER": "CPS_AVERAGE",
}
# this maps the geo logged colours to the codes
# taken from 'data/ASDCarleyBore/A97345/App2_Energia_LibraryCodes_201302.pdf'

colors = {
    "B": "blue",
    "E": "grey",
    "G": "green",
    "H": "khaki",
    "K": "pink",
    "N": "black",
    "O": "orange",
    "P": "purple",
    "R": "red",
    "U": "umber",
    "W": "white",
    "Y": "yellow",
}
intensity = {"D": "dark", "L": "light", "M": "medium"}

tmp_geo = {}
# loop over the each file and search for the .txt extension.
# look in the file name to see where to allocate the data
for i in outpath.glob("**/*.txt"):
    tmp, tmp_head = readers.dmp(i)
    if i.name.find("ASS") >= 0:
        tmp_assay.append(tmp)
    elif i.name.find("LITH") >= 0:
        tmp_lith.append(tmp)
    elif i.name.find("SCINT") >= 0:
        tmp.rename(columns=column_map, inplace=True)
        tmp_scint.append(tmp)
    elif (i.name.find("COLL") >= 0) and (i.parts[2] == "A108383"):
        tmp_collar.append(tmp)
    elif i.name.find("XRF") >= 0:
        tmp.columns = [n.replace("_HHXppm", "") for n in tmp.columns.to_list()]
        tmp.rename(columns=column_map, inplace=True)
        tmp_xrf.append(tmp)
    elif (i.name.find("GEO") >= 0) or (i.name.find("ALT") >= 0):
        tmp_geo.update({i.stem: tmp})
    else:
        pass

# concatenate the data
assay = pd.concat(tmp_assay)
lith = pd.concat(tmp_lith)
scint = pd.concat(tmp_scint)
collar = pd.concat(tmp_collar)

color = tmp_geo["CB_WADL4_GEO2015A_1"]  # colour
grain = tmp_geo["CB_WADL4_GEO2015A_2"]  # grain size
alter = tmp_geo["CB_WADL4_ALTE2015A"]  # alteration
strat = tmp_geo["CB_WADL4_GEO2015A"]  # stratigraphy
pxrf = pd.concat(tmp_xrf)
# rename the columns to the required format
# fix up the numeric columns:
column_map = {
    "Hole_id": "holeid",
    "HOLEID": "holeid",
    "DEPTH_FROM": "depthfrom",
    "DEPTH_TO": "depthto",
    "Easting_MGA": "easting",
    "Northing_MGA": "northing",
    "Elevation": "elevation",
    "Total Hole Depth": "depth",
    "Dip": "inclination",
    "Azimuth_TRUE": "azimuth",
    "STRATIGRAPHY": "strat",
    "Depth From": "depthfrom",
    "Depth To": "depthto",
}
numerics = [
    "depthfrom",
    "depthto",
    "depth",
    "easting",
    "northing",
    "elevation",
    "inclination",
    "azimuth",
    "CPS_AVERAGE",
]
assay_columns = [
    "Ca_ppm",
    "Fe_ppm",
    "K_ppm",
    "Mo_ppm",
    "P_ppm",
    "Ti_ppm",
    "Pb_ppm",
    "S_ppm",
    "Se_ppm",
    "Th_ppm",
    "V_ppm",
    "U_ppm",
    "U",
    "Th",
    "Pb",
    "Ca",
    "Fe",
    "K",
    "Mn",
    "Mo",
    "P",
    "Ti",
    "S",
    "Se",
    "V",
    "Co",
    "Cu",
    "Zn",
]
for i in [assay, lith, scint, collar, color, grain, alter, strat, pxrf]:
    i.rename(columns=column_map, inplace=True)
    iidx = i.columns.isin([*numerics, *assay_columns])
    num_cols = i.columns[iidx]
    i[num_cols] = i[num_cols].apply(pd.to_numeric, errors="coerce")
    # if there are assay columns replace the -ve values with nan
    aidx = i.columns.isin(assay_columns)
    if any(aidx):
        for c in i.columns[aidx]:
            idx = i[c] < 0
            i.loc[idx, c] = np.nan

from importlib import reload

reload(drillhole)
self = drillhole.DrillData(collar, strat=strat, assay=assay, negative_down=False)
mb = self.to_vtk()
mbvtks = pv.merge(mb["strat"])
p = pv.Plotter()
p.add_mesh(mbvtks.scale([1, 1, 50]), scalars="strat", line_width=10, cmap="tab20")
p.enable_fly_to_right_click()
p.show()


# bounds = np.asarray(mbvtks.bounds).reshape(3,2).astype(int).T
bounds = np.asarray([[293000, 7392000, -93], [298000, 7404000, 236]])
contacts, strat_column, dd = self.to_loop(
    fractional_depth=False, use_graph_estimate=True
)
contacts[["interface", "nx", "ny", "nz"]] = [np.nan, 0, 0, -1]

# model only FG
cover_index = (contacts.feature_name.str.find("Cz") == 0) & (
    contacts["type"] == "contact"
)
inside_index = (contacts.feature_name.str.find("Cz") == 0) & (
    contacts["type"] == "inside"
)
# contacts.loc[~cover_index,'feature_name'] = 'strati2'

top_interface = contacts.loc[cover_index].copy()
contacts.loc[contacts["type"]=='bottom','val'] = contacts[contacts["type"]=='bottom'].feature_name.map(dd).values
contacts.loc[cover_index, "interface"] = 0
top_interface["interface"] = 0

data = pd.concat([contacts])
# convert to non-conformable
data = data.reset_index(drop=True)
#data = data[(data.type == 'bottom')]
# make conformable layers
ss = {'strati':['Cz'],'strati1':["Kw", "Km", "Kb", "D", "Ba"]}
new_strat = {}
for n,i in enumerate(ss):
    idx = data.feature_name.isin(ss[i])
    data.loc[idx,'feature_name'] = i
    tmp_strat = {}
    for s in ss[i]:
        tmp_strat.update({s:{'min':dd[s]-1,'max':dd[s],'id':dd[s],'colour':np.random.rand(3, 1).ravel()}})
    new_strat.update({i: tmp_strat})


p =pv.Plotter()
p.add_mesh(sp.scale([1,1,20]),cmap='tab20')
p.enable_fly_to_right_click()
p.show()

data = data[data['type'].isin(['inside','bottom'])]
del gm
gm = GeologicalModel(bounds[0, :], bounds[1, :])
gm.data = data.drop(columns="type")
# loopstructural the definition of stratigraphy is conformable units
# so for an unconformity we need two stratigraphic units
gm.create_and_add_foliation('strati')
gm.add_unconformity(gm['strati'],0)
gm.create_and_add_foliation('strati1')
gm.set_stratigraphic_column(new_strat)

gm.update(verbose=True)

gm.get_stratigraphic_surfaces('strati1',bottoms=False)
gm.get_feature_by_name('strati1').surfaces(2)

p = pv.Plotter()
for i in gm.get_feature_by_name('strati').surfaces([0,1,2,3,4,5]):
    p.add_mesh(i.vtk().scale((1,1,50)))
p.show()

gm.get_stratigraphic_surfaces()
p.add_mesh(mbvtks.scale([1, 1, 50]), scalars="strat", line_width=10, cmap="tab20")

p = pv.Plotter()
surfs = gm.get_feature_by_name('strati').surfaces([0,1,2,3,4,10])
for i in range(6):
    p.add_mesh(surfs[i].vtk().scale((1, 1, 50)))
p.add_mesh(pp.scale((1, 1, 50)))
p.enable_fly_to_right_click()
p.show()



tmp_unconformable = []
for i in dd:
    if i == "Cz":
        ftype = "bottom"
    else:
        ftype = "top"
    didx = (data.feature_name == i) & (data.type == ftype)
    ubhid = data[didx].bhid.unique()
    for bhid in ubhid:
        bidx = data.bhid == bhid
        tmp = data[bidx].copy()
        new_z = tmp["Z"] - data.loc[didx & (data.bhid == bhid), "Z"].values
        tmp["val"] = new_z
        # find the 3 points on the boundary
        vneg = tmp["val"] < 0
        vpos = tmp["val"] > 0
        pmin = tmp["val"][vneg].argmax()
        pmax = tmp.loc[vpos, "val"].argmin()
        pos_min = vneg[vneg].index[pmin]
        pos_max = vpos[vpos].index[pmax]
        pzero = np.where((didx & (data.bhid == bhid)))[0]
        #
        # to = tmp.loc[[pos_min,pzero[0],pos_max]].copy()
        to = pd.concat([tmp.loc[pzero]] * 3).copy()
        to = to.reset_index(drop=True)
        to.loc[[0, 2], "val"] = [1, -1]
        to.loc[[0, 2], "Z"] = to.loc[[0, 2], "Z"] + [1, -1]
        to.feature_name = i
        to["type"] = "unconformable"
        tmp_unconformable.append(to)

data = pd.concat(tmp_unconformable)
pp = pv.PointSet(data.loc[data.feature_name == "Cz", ["X", "Y", "Z"]])
pp["scalar"] = data.loc[data.feature_name == "Cz", ["val"]]
p = pv.Plotter()
p.add_mesh(pp, scalars="scalar")
p.enable_fly_to_right_click()
p.show()

gm = GeologicalModel(bounds[0, :], bounds[1, :])
gm.data = data.drop(columns="type")

# loopstructural the definiation of stratigraphy is conformable units
# so for an unconformity we need two stratigraphic units
new_strat = {}
for n, i in enumerate([["Cz"], ["Kw", "Km", "Kb", "D", "Ba"]]):
    tmp_strat = {dd[k]: strat_column[dd[k]] for k in i}
    new_strat.update({f"strati{n+1}": tmp_strat})

for i in dd:
    gm.create_and_add_foliation(
        i, nelements=30e2, interpolatortype="FDI", regularization=1
    )

gm.update(verbose=True)


p = pv.Plotter()

p.add_mesh(mbvtks.scale([1, 1, 50]), scalars="strat", line_width=10, cmap="tab20")
for i in dd:
    surfs = gm.get_feature_by_name(i).surfaces([0])
    p.add_mesh(surfs[0].vtk().scale((1, 1, 50)))
p.enable_fly_to_right_click()
p.show()

p = pv.Plotter()
p.add_mesh(mbvtks.scale([1, 1, 50]), scalars="strat", line_width=10, cmap="tab20")
for i in dd.keys():
    try:
        p.add_mesh(
            gm.get_feature_by_name(i).surfaces(value=dd[i])[0].vtk().scale((1, 1, 50))
        )
    except Exception:
        pass

p.enable_fly_to_right_click()
p.show()
