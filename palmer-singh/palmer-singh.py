# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import utilities.data_utils as dus
import utilities.preprocessing as pp
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.constants as mtc
from moist_thermodynamics import saturation_vapor_pressures as svp

cw = 165 / 25.4
hcw = 80 / 25.4

# %% 
# - Prepare Datasets
cids = dus.get_cids()
datasets = {
    "rapsodi": dus.open_radiosondes(cids["radiosondes"]),
    "beach": dus.open_dropsondes(cids["dropsondes"]),
    "gate": dus.open_gate(cids["gate"]),
}

for name, ds in datasets.items():
    datasets[name] = (
        ds.pipe(pp.interpolate_gaps).pipe(pp.extrapolate_sfc).pipe(pp.sel_percusion_E)
    )

datasets["orcestra"] = xr.concat(
    [datasets["rapsodi"], datasets["beach"]],
    dim="sonde",
)
#%%
# - get mean height of 850 hPa and 500 hPa levels
hgts = []
for P in [85000,50000]:
    dP = (ds.p - P)**2
    Z = ds['altitude'].broadcast_like(ds['p']).where(dP == dP.min(dim='altitude'), drop=True)
    hgts.append(Z.mean().values)
dz = hgts[1]-hgts[0]
#%%
ds = datasets["orcestra"]

mse = mtf.make_static_energy((mtc.lv0 + mtc.cl *mtc.T0))
es = svp.liq_murphy_koop  
qs = mtf.partial_pressure_to_specific_humidity(es(ds.ta),ds.p)

def mse_simple(T,Z,q):
    cp = mtc.cpd + (mtc.cl-mtc.cpd)*q
    a  = cp*T + Z*mtc.gravity_earth + mtf.vaporization_enthalpy(T)*q
    b  = mtc.cpd*T + Z*mtc.gravity_earth + mtc.lv0*q
    return a

hs_exact  = mse(ds.ta,ds.altitude,qs)
hs_simple = mse_simple(ds.ta,ds.altitude,qs)

hs = hs_exact
dhs = hs.sel(altitude=5777, method='nearest') - hs.sel(altitude=1517, method='nearest')
dqs = (qs - ds.q).sel(altitude=slice(hgts[0],hgts[1])).mean(dim='altitude')
x = -dhs/dqs/mtc.lv0/dz*1000.


sns.set_context("paper")
fig, ax = plt.subplots(1, 1, figsize=(hcw, hcw))
x.plot.hist(ax=ax,bins=100,range=(-0.1,2.1))
plt.xlabel('$\\varepsilon$ / km$^{-1}$')
plt.ylabel('Frequency')
sns.despine(offset=10)
plt.show()

#%%
# plot 2D histogram
eps = 0.5e-3
heps = np.arange(-20,-3,0.1)
qeps = -heps/dz/eps

sns.set_context("paper")
fig, ax = plt.subplots(1, 1, figsize=(hcw , hcw ))
sns.kdeplot(ax = ax, x=dqs*mtc.lv0/1000, y=dhs/1000, levels=7, cmap='Blues', cbar=False)
plt.plot(qeps,heps,label=f'$\\varepsilon = {eps*1000}$ km $^{{-1}}$')
ax.set_xlabel('$\\frac{\\ell_\\mathrm{v}}{\\Delta z} \\int (q_\\mathrm{s} - q)\\, \\mathrm{d}z$ /  kJkg$^{-1}$')
ax.set_ylabel('$\\Delta h_s$ /  kJkg$^{-1}$')
ax.set_ylim(-25,-3)
ax.set_xlim(0,13)
sns.despine(offset=10)
plt.legend()
plt.show()

# %%
