


import math
import shutil
from importlib import reload
# label prop in 3d space for geology is easily done by first triangulating the collars
# to create a spatial join then after that for this scenario we are going to do a simple sequential 
# numbering of litho types
# label prop in 3d space for geology is easily done by first triangulating the collars
# to create a spatial join then after that for this scenario we are going to do a simple sequential 
# numbering of litho types
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.approximation as nx_app
import numpy as np
import pandas as pd
import pyvista as pv
from augeosciencedatasets import downloaders, readers
from fastdtw import fastdtw
from LoopStructural import GeologicalModel
from LoopStructural.datatypes import BoundingBox
from LoopStructural.visualisation import Loop2DView, Loop3DView
from matplotlib import colors
from matplotlib import pyplot as plt
from networkx.algorithms import approximation
from pyvista import (LineSource, MultipleLines, line_segments_from_points,
                     lines_from_points)
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, Delaunay, delaunay_plot_2d
from scipy.spatial.distance import cdist, cityblock, cosine, pdist, squareform
from sklearn.linear_model import RANSACRegressor
# calculate distance matrices fast
from sklearn.neighbors import KDTree
from tqdm import tqdm

import drillhole
from dipconversion import strikedip2vector
from drillhole import Drillhole

reload(drillhole)
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
results["geop"].rename(columns={'Depth':'depth'},inplace=True)
results["struct"].rename(columns={'Depth':'depth','Dip direction':'dipdirection','Dip':'dip'},inplace=True)
results['assay'].replace(to_replace=[-999,-100],value=np.nan,inplace=True)
lith_codes = pd.read_csv('data/A115077/Drilling/TH_GEOL_CODES.txt',sep='\t+',engine='python')

lithcats =pd.Categorical(results['geo'].Lithology).codes
tmp =[]
for i in ['THRC276','THRC281']:
    idx = results['geo'].HOLEID == i

    tmp.append(lithcats[idx].astype(float))



collars = results["collars"].copy()
strat = results['geo'].copy()


tri = Delaunay(collars[['easting','northing']])

plt.triplot(collars.easting,collars.northing,tri.simplices)
plt.plot(collars.easting,collars.northing,'.')
plt.show()

# label prop in 3d space for geology is easily done by first triangulating the collars
# to create a spatial join then after that for this scenario we are going to do a simple sequential 
# numbering of litho types
#for i in tri.simplices:
#    tmpc =[]
#    for j in collars.loc[i,'HOLEID'].values.tolist():
#        gidx = results['geo'].HOLEID ==j
#        results['geo'][gidx]
#


dev = {}
deva = {}

results['struct']
i = 'THRC326'
scopy = results["surv"].copy()
scopy.inclination = 90+scopy.inclination
td = []
tdn = []
tvtk = []
dhs = {}

for i in tqdm(results["collars"].HOLEID.unique()):
    tmp_surv = extract_hole(scopy, i)
    tmp_ass = extract_hole(results["assay"], i)
    tmp_col = extract_hole(results["collars"], i)
    tmp_geop = extract_hole(results["geop"], i)
    tmp_geol = extract_hole(results["geo"], i)
    tmp_struct = extract_hole(results["struct"], i)

    # lith we assume is strat here
    tmp_strat = tmp_geol.iloc[:,:4].copy().rename(columns={'Lithology':'strat'})
    # assume that last OB material OB to there

    tmp_strat.replace(simp_strat,inplace=True)
    whereob = np.where(tmp_strat.strat == 'OB')
    if len(whereob[0])>0:
        tmp_strat.loc[0:whereob[0].max(),'strat'] = 'OB'
    
    # simplify the overburden materials
    tmpdh = Drillhole(i,tmp_col.depth, 30,300,tmp_col.easting,tmp_col.northing, tmp_col.elevation,assay=tmp_ass,geology=tmp_geol,survey=tmp_surv,strat=tmp_strat,geophysics=tmp_geop,struct=tmp_struct,strat_simplify=simp_strat)
    dhs.update({i:tmpdh})

    tdata = tmpdh.create_vtk()

    td.append(tdata)


strikedip2vector(tdata['dataname'])
vvars= ['assay', 'geology', 'strat', 'geophysics', 'watertable','struct']

bins = {v:[] for v in vvars}
bnames = {v:[] for v in vvars}
for n,i in enumerate(td):
    for v in i.keys():
        bins[v].append(i[v])
        
txyz = []
tdf = []
for i in bins['struct']:
    txyz.append(i['points'])
    tdf.append(i['data'])
i['data']
results['struct']['Structure type'].unique()
txy = np.concatenate(txyz)
tp = pd.concat(tdf)
dm = []

for p, n in zip(txy,strikedip2vector(tp.dipdirection,tp.dip)):
    dm.append(pv.Disc(p,inner=0,normal=n,outer=10))

pv.merge(dm).plot()

blockout = []
i = 'geophysics'
multiblocks = {}
for i in vvars:
    tmpbins = bins[i]
    if len(tmpbins)>0:
        mb = pv.MultiBlock(tmpbins)
        mb = mb.as_polydata_blocks()
        for n,name in enumerate(bnames[i]):
            mb.set_block_name(n,name)
        multiblocks.update({i:mb})
        mb.save(f'vtk/{i}.vtmb')



tmp_contacts = []
for i in dhs:
    if isinstance(dhs[i].strat,pd.DataFrame):
        tmpc = dhs[i].contacts.copy()
        tmpc['bhid'] = i
        tmp_contacts.append(tmpc)


contacts = pd.concat(tmp_contacts)

contacts = contacts[~contacts.feature_name.isin(['','UNK','NSR'])].reset_index(drop=True)

# calculation for better insides 
idxfrac = contacts.type=='inside'
contacts[idxfrac]
max_depth = contacts[idxfrac].groupby('feature_name')['val'].max()
contacts[idxfrac].feature_name.unique()
frac_depth = contacts.val/contacts.feature_name.map(max_depth.to_dict())


strat_order = contacts[idxfrac].groupby('feature_name')['Z'].max().reset_index().sort_values('Z',ascending=False).reset_index(drop=True)
strat_map = {}
for n,i  in strat_order.iterrows():
    strat_map.update({i.feature_name:n})

contacts.loc[idxfrac,'val'] = contacts[idxfrac].feature_name.map(strat_map)+frac_depth[idxfrac]


bounds = np.asarray(multiblocks['assay'].bounds).reshape(3,2).astype(int).T
pv.merge(multiblocks['strat'])['strat']


ctmp = contacts[contacts.type == 'inside'].copy()
# relabel datacollars = results['collars'][results['collars'].HOLEID.isin(tops.bhid)]
collars = collars.reset_index(drop=True)

tri = Delaunay(collars[['easting','northing']])


xyzn = ['X','Y','Z']

u = 'MG'
fidx = ctmp.feature_name == u
collars = collars.reset_index(drop=True)

tri = Delaunay(collars[['easting','northing']])
# collar indices
tz = []
for i in tri.simplices:
    tz.append(collars.iloc[i].HOLEID.values)

G = nx.Graph()
for path in tri.simplices:
    point_dist = pdist(tri.points[path])
    for p in range(1,3):
        nx.add_path(G, [path[0],path[p]],weight= point_dist[p-1])
Gt = nx.minimum_spanning_tree(G,)

plt.spy(nx.adjacency_matrix(Gt))
plt.show()

cdist(np.asarray([[ 636788.09, 6440770.91]]),np.asarray([[ 636799.29, 6440769.83]]))
u = 'MG'
fidx = ctmp.feature_name == u

pedge = nx.to_edgelist(Gt)

# collar indices
tz = []
for i in pedge:
    tz.append(collars.iloc[np.asarray(i[0:2])].HOLEID.values)

nn = len(tz)
cm= mpl.colormaps['viridis']
tz[0]
for n,i in enumerate(tz):
    idx1 = collars.HOLEID.isin(i)
    plt.plot(collars[idx1].easting.T,collars[idx1].northing.T,color=cm(n/nn)) 
plt.show()



def angle(p1, p2):
    dxy = p1-p2
    return np.rad2deg(np.arctan(dxy[0]/dxy[1]))+180


XY = tri.points
all_selected = []
kdt = KDTree(XY)
centre = XY.mean(0).reshape(-1,2)

def closest_no_masking(kdt,XY, node,angle_tolerance = 5):
    '''
    finds the n closest points to a kdt we only return points that are not masked by another
    i.e. the points cannot have the same direction as another point. 
    '''
    d, v = kdt.query(XY[node,:].reshape(-1,2),k=20)
    # calculate point angles to centre point
    pangle = cdist(XY[node].reshape(-1,2),XY[v.ravel()],angle)
    # find where there are more than 2 counts in a bin the further samples are masked from the origin by another
    pbins = np.digitize(pangle.ravel(), np.arange(0,361,angle_tolerance))
    ub, cb = np.unique(pbins,return_counts=True)
    mc = ub[cb>1]
    selected_points = []
    for i in mc:
        idx= pbins == i
        pmin = d.ravel()[idx].argmin()
        tmp = np.where(idx)[0][pmin]
        selected_points.append(tmp)
    selected_points.extend(np.where(cb ==1)[0].tolist())
    selected_points = np.asarray(selected_points)
    final_index = v.ravel()[selected_points]
    final_dist = d.ravel()[selected_points]
    outindex = final_index !=node
    final_dist = final_dist[outindex]
    final_index = final_index[outindex]
    return final_index,final_dist

# find the closest point to the centre
_, starting_node = kdt.query(centre,k=1)

# lets just loop over all the nodes
starting_node = starting_node.item()
final_index = closest_no_masking(kdt, XY, starting_node)
# lets just loop over all the nodes
gdict = {}
ga = nx.Graph()
for i in  range(len(XY)):
    final_index,final_dist = closest_no_masking(kdt, XY, i,angle_tolerance=0.1)
    for fi,fd in zip(final_index,final_dist):
        ga.add_edge(i,fi,weight=fd)
        gdict.update({i:fi,'weight':fd})


for i in nx.bfs_edges(ga,starting_node):
    plt.plot(XY[list(i)].T[0,:],XY[list(i)].T[1,:],'-')
plt.show()


edge_list = list(nx.bfs_edges(G,starting_node))
from tslearn import metrics
from scipy.optimize import minimize
tmp_feat = ctmp.copy()
tmp_feat =tmp_feat.replace(to_replace={'GMG':'MG','FMV':'FG','LGR':'FG','GRA':'FG'})
fnum = pd.Categorical(tmp_feat.feature_name).codes
tmp_feat['strat_num'] = np.nan
tmp_feat['fnum'] = fnum
singleStrat = 'MG'

# let's attempt group wise matching
for l in nx.bfs_layers(G, starting_node):
    c1 = collars.iloc[starting_node].HOLEID
    idx1 = (tmp_feat.bhid == c1)
    c2 = collars.iloc[l].HOLEID
    idx2 = (tmp_feat.bhid.isin( c2)) 
    tdm = cdist(tmp_feat[idx1][xyzn],tmp_feat[idx2][xyzn])
    RANSACRegressor().fit(tmp_feat[idx2][xyzn[0:2]])

'''
 this is the best working version
'''

tmp_feat = ctmp.copy().reset_index(drop=True)
tmp_feat =tmp_feat.replace(to_replace={'GMG':'MG','FMV':'FG','LGR':'FG','GRA':'FG'})
fnum = pd.Categorical(tmp_feat.feature_name).codes
tmp_feat['strat_num'] = np.nan
tmp_feat['fnum'] = fnum

constrainSingle = False 
maskStrats  = False
doDTW = True
singleStrat = 'MG'
unlabelled = True
useThickness = True
# this is a pretty shit algo pairwise matching
for n,i in enumerate(edge_list[0:10]):
    c1 = collars.iloc[i[0]].HOLEID
    c2 = collars.iloc[i[1]].HOLEID
    if singleStrat == False:
        idx1 = (tmp_feat.bhid == c1)
        idx2 = (tmp_feat.bhid == c2)
    else:
        idx1 = (tmp_feat.bhid == c1) & (tmp_feat.feature_name == singleStrat)
        idx2 = (tmp_feat.bhid == c2) & (tmp_feat.feature_name == singleStrat)

    if idx1.any() and idx2.any():        
        # label the first successful result
        if unlabelled:
            nunits = idx1.sum()
            tmp_feat.loc[idx1,'strat_num'] = np.arange(nunits)
            # with the first node labelled don't execute this anymore
            unlabelled = False
        
        tdm = cdist(tmp_feat[idx1][xyzn],tmp_feat[idx2][xyzn])
        if maskStrats:
            tda = cdist(tmp_feat[idx1][['fnum']],tmp_feat[idx2][['fnum']],'cityblock')
            dmask = tda !=0
            tdm[dmask] = tdm[dmask]*10
        if useThickness:
            tda = cdist(tmp_feat[idx1][['val']],tmp_feat[idx2][['val']])
            tdm+=tda

        # find the best 1:1 mapping
        if doDTW:
            tp,dm = metrics.dtw_path_from_metric(tdm,metric='precomputed')
            tp = np.asarray(tp)
            ia,ib = tp[:,0], tp[:,1]
        else:
            ia, ib = linear_sum_assignment(tdm)
        # the best 1:1 mapping doesn't always have a single line fit between drillholes
        # cause holes are drilled through different sections of the orebody we shouldn't expect 
        # that this will work all the time, hence we will further constrain the output to only 
        # look for combinations that have a single slope
        if constrainSingle:
            tdiff = np.diff(ia-ib)
            diff_num = np.abs(tdiff)>0
            pdiff = np.where(diff_num)[0]
            if len(pdiff):
                splita = np.array_split(ia,pdiff+1)
                splitb = np.array_split(ib,pdiff+1)
                longest = np.argmax([len(i) for i in splita])
                ia = splita[longest]
                ib = splitb[longest]
            # check if there are stratnumber on either of them if so renumber where appropriate
        indexa = tmp_feat[idx1].index[ia]
        indexb = tmp_feat[idx2].index[ib]
        strat_num = tmp_feat.iloc[indexa].strat_num.values
        # prevent stratnums from going to nans
        nanidx = np.isnan(strat_num)
        if any(nanidx):    
            mstrat = np.nanmin(strat_num)
            strat_num = np.arange(len(strat_num))-mstrat

        tmp_feat.loc[indexb,'strat_num'] = strat_num


plt.plot(tmp_feat[~tmp_feat.strat_num.isna()].strat_num.values)
plt.show()

p= pv.Plotter()
for i in range(0,20):
    tp = pv.PointSet(tmp_feat[tmp_feat.strat_num == i][xyzn])
    if tp.n_points > 3:
        trip = tp.cast_to_polydata().delaunay_2d()
        p.add_mesh(trip,opacity=0.2)
for i in tmp_feat[~tmp_feat.strat_num.isna()].bhid.unique():
    idx = tmp_feat.bhid == i

    pp = pv.PointSet(tmp_feat.loc[idx,xyzn])
    if ~tmp_feat[idx].feature_name.isna().all():
        pp['scalars'] = tmp_feat[idx].feature_name
        tmp.append(pp)
p.add_mesh(pv.merge(tmp),scalars='scalars',cmap='tab20')
p.show()

clean_feat = tmp_feat[~tmp_feat.strat_num.isna()].copy()
clean_feat['val'] = clean_feat['strat_num']

gm = GeologicalModel(bounds[0,:],bounds[1,:])
clean_feat.feature_name = 'strati'
# model only FG
gm.data = clean_feat.drop(columns='type') 
gm.create_and_add_foliation('strati',nelements=20e2,interpolatortype='PLI',regularisation=10)
tmdict = {}
for i in range(int(clean_feat.val.min()),int(clean_feat.val.max())):
    tmdict.update({i:{'min':i,'max':i+1,'id':i}})

strat_column = {'strati':tmdict}
gm.set_stratigraphic_column(strat_column)
gm.update(verbose=True)

surfs = gm.get_stratigraphic_surfaces()
fn= gm.evaluate_model(clean_feat[xyzn])
plt.hist(fn-clean_feat.val,bins=np.arange(-10.5,10.5))
plt.show()

p = pv.Plotter()
for i in surfs[::1]:
    p.add_mesh(i.vtk())

for i in tmp_feat[~tmp_feat.strat_num.isna()].bhid.unique():
    idx = tmp_feat.bhid == i
    pp = pv.PointSet(tmp_feat.loc[idx,xyzn])
    pp['scalars'] = tmp_feat[idx].strat_num
    p.add_mesh(pp,cmap='viridis')
p.show()



ufeat = ctmp.feature_name.unique()
for u in ufeat:
dmat = cdist(tmp_feat[xyzn],tmp_feat[xyzn])
# masking do adjacency matrix

np.isfinite(dmat).sum()

bdist = cdist(pd.Categorical(tmp_feat['bhid']).codes.reshape(-1,1),pd.Categorical(tmp_feat['bhid']).codes.reshape(-1,1))
mask = bdist==0
dmat[mask] = np.inf
dmat[nx.to_numpy_array(Gt)==0]=np.inf
# search max
dmat[dmat>100] = np.inf
dmat[np.triu_indices_from(dmat)] = np.inf
plt.spy(np.isfinite(dmat) == 1)
plt.show()
ik = []
i = tz[50]
for i in tz:
    idx1 = tmp_feat.bhid == i[0]
    idx2 = tmp_feat.bhid == i[1]
    # indexing tricks to get the position in the array
    if idx1.any() and idx2.any():
        ia, ib = np.ix_(idx1,idx2)
        r, c = np.unravel_index(dmat[ia,ib].argmin(),(idx1.sum(),idx2.sum()))
        pr, pc = ia.ravel()[r],ib.ravel()[c]
        # mask out the rest of the array so that any other point in those holes can no longer be selected
        ma = ia[ia!=pr]
        mb = ib[ib!=pc]
        dmat[ma.reshape(-1,1), mb.reshape(1,-1)] = np.inf
        ik.append(tmp_feat.iloc[pr])
        ik.append(tmp_feat.iloc[pc])
pd.concat(ik,axis=1).T.drop_duplicates()[xyzn]

X = pd.concat(ik,axis=1).T
ins = RANSACRegressor().fit(X[["X","Y"]],X[["Z"]]).inlier_mask_
ppp = pv.PointSet(X[xyzn].astype(float))
ppp['scalar'] = ins
ppp.plot(scalars='scalar')
pv.PointSet(pd.concat(ik,axis=1).T.drop_duplicates()[xyzn].astype(float)).plot()

plt.imshow(dmat)
plt.show()

nn = len(tz)
cm= mpl.colormaps['viridis']
tz[0]
for n,i in enumerate(tz):
    idx1 = collars.HOLEID.isin(i)
    plt.plot(collars[idx1].easting.T,collars[idx1].northing.T,color=cm(n/nn)) 
plt.show()

ctmp.feature_name
pp = pv.PointSet(ctmp[['X','Y','Z']])
pp['scalars'] = ctmp.feature_name
pp.plot(scalars='scalars',colormap='tab20',background='grey')


ctmp.feature_name = 'strati'
bounds = np.asarray(pv.PointSet(ctmp[['X','Y','Z']]).bounds).reshape(3,2).astype(int).T

gm = GeologicalModel(bounds[0,:],bounds[1,:])

# model only FG
gm.data = ctmp.drop(columns='type') 
gm.create_and_add_foliation( 'strati',nelements=20e4)
gm.add_unconformity(gm['strati'],1)


strat_column = {'strati':{i:{'min':strat_map[i],'max':strat_map[i]+1,'id':strat_map[i]} for i in strat_map}}
gm.set_stratigraphic_column(strat_column)

gm.update(verbose=True)
surfs = gm.get_stratigraphic_surfaces()

p = pv.Plotter()
for i in surfs:
    p.add_mesh(i.vtk())
p.show()



viewer = Loop3DView(gm,background='lightgrey')
viewer.plot_data(gm['strati'],pyvista_kwargs={'colormap':'tab20'})
viewer.plot_model_surfaces(gm['strati'])
viewer.show(interactive=True)

for i in surfs:
    viewer.add_mesh(i.vtk(),opacity=0.1)



# label propagation tricks
tops = contacts[contacts['type'] == 'top'].copy()
collars = results['collars'][results['collars'].HOLEID.isin(tops.bhid)]
collars = collars.reset_index(drop=True)

tri = Delaunay(collars[['easting','northing']])


xyzn = ['X','Y','Z']

for i in tri.simplices:
    tz = []

    for j in collars.loc[i,'HOLEID'].values.tolist():
        if hasattr(dhs[j],'contacts'):  
            tz.append(dhs[j].contacts)
    if len(tz)>1:
        break

    t = tz[0]
    for c1 in range(1,len(tz)):
        t1 = tz[c1]
        for tt in ['bottom','top','contact','inside']:
            # xyz distance
            tmp1 = t[t['type'] == tt]
            tmp2 = t1[t1['type'] == tt]
            dm = cdist(tmp1[xyzn],tmp2[xyzn])
            
            tmp1,tmp2
            for n,v in tmp1.iterrows():
                twhere = np.where(tmp2.feature_name == v.feature_name)[0]
                dm[n,twhere]
            sf = set(tmp1.feature_name)
            sf.update(tmp2.feature_name)
            sf = list(sf)

            lm = cdist(pd.Categorical(tmp1.feature_name,categories=sf).codes.reshape(-1,1),pd.Categorical(tmp2.feature_name,categories=sf).codes.reshape(-1,1),'canberra')
            plt.imshow(lm+dm/dm.max())
            plt.show()
            # for all the lith types in the first find the closest inspace
            dist, steps = fastdtw(pd.Categorical(tmp1['feature_name']).codes.reshape(-1,1),pd.Categorical(tmp2['feature_name']).codes.reshape(-1,1),dist=cityblock)
            np.asarray(steps)
            plt.plot(np.asarray(steps)[:,0],np.asarray(steps)[:,1])
            plt.show()
            tmp2.loc[11]
            tmp1.loc[9]

tt = 'bottom'
plt.triplot(collars.easting,collars.northing,tri.simplices)
plt.plot(collars.easting,collars.northing,'.')
plt.show()

tops.groupby('bhid')['Z'].min()
pv.PointSet(tops[['X','Y','Z']]).plot(scalars=tops['feature_name'],cmap='tab20')
