import pandas as pd
import numpy as np
from augeosciencedatasets import readers
from augeosciencedatasets import downloaders
from drillhole import Drillhole
from matplotlib import pyplot as plt
from pathlib import Path
import shutil

outpath = Path("data/A115077")
paths = pd.read_csv("examples/A115077.csv")
if not outpath.exists():
    outpath.mkdir(parents=True)

for _, p in paths.iterrows():
    outfile = outpath.joinpath(p["FileName"])
    downloaders.from_dasc(p["URL"], outfile)


shutil.unpack_archive(outpath.joinpath("Drilling.zip"), outpath.joinpath("Drilling"))


files = {
    "collars": {
        "file": "TH_WASL4_COLL2017A.txt",
        "numerics": ["Easting_MGA", "Northing_MGA", "Elevation", "Total Hole Depth"],
    },
    "geo": {"file": "TH_WADL4_GEO2017A.txt", "numerics": ["Depth From", "Depth To"]},
    "assay": {
        "file": "TH_WADG4_ASS2017A_RC.txt",
        "numerics": [
            "From",
            "To",
            "INTERVAL",
            "Au",
            "Au-Rp1",
            "Au-Rp2",
            "Ag",
            "As",
            "As_Rp1",
            "Cu",
            "Fe",
            "Pb",
            "S",
            "Zn",
            "Ag-pXRF",
            "Al-pXRF",
            "As-pXRF",
            "Au-pXRF",
            "Ba-pXRF",
            "Bal-pXRF",
            "Bi-pXRF",
            "Br-pXRF",
            "Ca-pXRF",
            "Cd-pXRF",
            "Ce-pXRF",
            "Cl-pXRF",
            "Co-PXRF",
            "Cr-pXRF",
            "Cu-pXRF",
            "Fe-pXRF",
            "Hg-pXRF",
            "K-pXRF",
            "Mg-pXRF",
            "Mn-pXRF",
            "Mo-pXRF",
            "Nb-pXRF",
            "Nd-pXRF",
            "Ni-pXRF",
            "La-pXRF",
            "P-pXRF",
            "Pb-pXRF",
            "Pd-pXRF",
            "Pr-pXRF",
            "Pt-pXRF",
            "Rh-pXRF",
            "Rb-pXRF",
            "S-pXRF",
            "Sb-pXRF",
            "Se-pXRF",
            "Si-pXRF",
            "Sn-pXRF",
            "Sr-pXRF",
            "Ta-pXRF",
            "Th-pXRF",
            "Ti-pXRF",
            "U-pXRF",
            "V-pXRF",
            "W-pXRF",
            "Y-pXRF",
            "Zn-pXRF",
            "Zr-pXRF",
        ],
    },
    "struct": {
        "file": "TH_WADL4_STRU2017A.txt",
        "numerics": ["Depth", "Dip direction", "Dip", "Aperture"],
    },
    "surv": {
        "file": "TH_WADS4_SURV2017A.txt",
        "numerics": ["Surveyed Depth", "Azimuth_TRUE", "Dip"],
    },
    "geop": {
        "file": "TH_WADL4_DLOG2017A.txt",
        "numerics": ["Depth", "CAL", "CDL", "NGAM", "MSUS", "MagnField"],
        "nan": "-9999",
    },
}

results = {}
for f in files:
    tmp_file = outpath.joinpath("Drilling").joinpath(files[f]["file"])
    tmp_data, header = readers.dmp(tmp_file)
    # nan replace if required
    if "nan" in files[f]:
        tmp_data = tmp_data.replace(files[f]["nan"], np.nan)

    tmp_data[files[f]["numerics"]] = tmp_data[files[f]["numerics"]].apply(pd.to_numeric)
    results.update({f: tmp_data})


def remap_meta(x):
    names = {"HOLEID", "FROM", "TO"}
    y = x.upper().replace("_", "")
    if y in names:
        return y
    else:
        return x


def extract_hole(table, holeid):
    idx = table.HOLEID == holeid
    return table[idx].reset_index(drop=True).copy()



for i in results.keys():
    results[i].columns = [remap_meta(c) for c in results[i].columns]

results["collars"].rename(columns={'Easting_MGA':'easting','Northing_MGA':'northing','Elevation':'elevation','Total Hole Depth':'depth'},inplace=True)
# remap the column names to the correct ones
results['surv'].rename(columns={'Surveyed Depth':'depth','Azimuth_TRUE':'azimuth','Dip':'inclination'},inplace=True)
results['assay'].rename(columns={'FROM':'depthfrom','TO':'depthto'},inplace=True)
results['surv'].rename(columns={'Depth':'depth'},inplace=True)
results["geo"].rename(columns={'Depth From':'depthfrom','Depth To':'depthto'},inplace=True)
dev = {}
deva = {}

from pyvista import LineSource,MultipleLines
import pyvista as pv 
p = pv.Plotter()
for i in results["collars"].HOLEID:
    tmp_surv = extract_hole(results["surv"], i)
    tmp_ass = extract_hole(results["assay"], i)
    tmp_col = extract_hole(results["collars"], i)
    tmp_geop = extract_hole(results["geop"], i)
    tmp_geol = extract_hole(results["geo"], i)
    # lith we assume is strat here
    tmp_strat = tmp_geol.iloc[:,:4].copy().rename(columns={'Lithology':'strat'})
    tmpdh = Drillhole(i,tmp_col.depth, -60,300,tmp_col.easting,tmp_col.northing, tmp_col.elevation,assay=tmp_ass,survey=tmp_surv,geology=tmp_geol,positive_down=False)
    xyz,tdata = tmpdh.create_vtk()
    ll = MultipleLines(xyz)
    p.add_mesh(ll)
p.show()
tmpdh.type_map.keys()
tmpdh.x
tmpdh.y
LineSource(xyz)

from importlib import reload
from src import drillhole
reload(drillhole)