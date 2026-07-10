# %%

import xarray as xr
import utilities.data_utils as dus
import utilities.preprocessing as pp
import moist_thermodynamics.functions as mtf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.stats import norm

# %%
cids = dus.get_cids()
gatel3 = (
    dus.open_gate("QmWZryTDTZu68MBzoRDQRcUJzKdCrP2C4VZfZw1sZWMJJc")
    .pipe(pp.interpolate_gaps)
    .pipe(pp.extrapolate_sfc)
    .pipe(pp.sel_percusion_E)
)
beach = (
    dus.open_dropsondes(cids["dropsondes"])
    .pipe(pp.interpolate_gaps)
    .pipe(pp.extrapolate_sfc)
    .pipe(pp.sel_percusion_E)
)
rapsodi = (
    dus.open_radiosondes(cids["radiosondes"])
    .pipe(pp.interpolate_gaps)
    .pipe(pp.extrapolate_sfc)
    .pipe(pp.sel_percusion_E)
)

gatel3 = gatel3.assign(
    n2=xr.apply_ufunc(
        mtf.brunt_vaisala_frequency,
        gatel3.theta,
        gatel3.q,
        gatel3.altitude,
        input_core_dims=[["altitude"], ["altitude"], ["altitude"]],
        output_core_dims=[["altitude"]],
        vectorize=True,
    )
)
# %%
gate_new = dus.open_gate(cids["gate"]).pipe(pp.sel_percusion_E)
# %%
n2mean = mtf.brunt_vaisala_frequency(
    gatel3.theta.mean("sonde"),
    gatel3.q.mean("sonde"),
    gatel3.altitude,
)

# %%
gate_new.n2.mean("sonde").sel(altitude=slice(0, 12000)).coarsen(
    altitude=10, boundary="trim"
).mean().plot(y="altitude", label="mean of n2")
gatel3.n2.mean("sonde").sel(altitude=slice(0, 12000)).coarsen(
    altitude=10, boundary="trim"
).mean().plot(y="altitude", label="mean of n2")
n2mean.sel(altitude=slice(0, 12000)).coarsen(altitude=10, boundary="trim").mean().plot(
    y="altitude", label="n2 of mean"
)
# plt.axhline(9000, color="k", ls="--", lw=0.5)
plt.legend()
plt.ylabel("altitude / m")
plt.xlabel("Brunt-Vaisala frequency / s-1")
sns.despine()

# %%

std = gatel3.n2.sel(altitude=slice(8000, 10000)).max(dim="altitude").std()


ds = gatel3.n2.sel(altitude=slice(8000, 10000)).max(dim="altitude")
mean, std = norm.fit(ds.where(ds < 3 * std).dropna("sonde").values)


# %%
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

gatel3.n2.sel(altitude=slice(8000, 10000)).max(dim="altitude").plot.hist(
    ax=axes[0], bins=100, range=(0.01, 0.1)
)
gate_new.n2.sel(altitude=slice(8000, 10000)).max(dim="altitude").plot.hist(
    ax=axes[0], bins=100, range=(0.01, 0.1)
)
axes[0].axvline(mean + 3 * std, color="k", ls="--", lw=0.5)
xvals = np.linspace(0, 0.1, 100)
axes[0].plot(xvals, norm.pdf(xvals, mean, std))

cond = gatel3.n2.sel(altitude=slice(8000, 10000)).max(dim="altitude") < mean + 3 * std
gatel3.n2.mean("sonde").sel(altitude=slice(0, 12000)).coarsen(
    altitude=10, boundary="trim"
).mean().plot(ax=axes[1], y="altitude", label="all mean")
gatel3.where(cond).n2.mean("sonde").sel(altitude=slice(0, 12000)).coarsen(
    altitude=10, boundary="trim"
).mean().plot(ax=axes[1], y="altitude", label="cutoff mean")
gate_new.n2.mean("sonde").sel(altitude=slice(0, 12000)).coarsen(
    altitude=10, boundary="trim"
).mean().plot(ax=axes[1], y="altitude", label="new mean")
axes[1].legend()
axes[1].set_ylabel("altitude / m")
axes[1].set_xlabel("n2 / s-1")
axes[0].set_xlabel("max(n2) between 8000 and 10000 m / s-1")
(beach.ta.mean("sonde") - gatel3.ta.mean("sonde")).sel(
    altitude=slice(8000, 10000)
).plot(ax=axes[2], y="altitude", color="teal", alpha=0.5, ls=":", label="old")
(rapsodi.ta.mean("sonde") - gatel3.ta.mean("sonde")).sel(
    altitude=slice(8000, 10000)
).plot(ax=axes[2], y="altitude", color="navy", alpha=0.5, ls=":")
(beach.ta.mean("sonde") - gate_new.ta.mean("sonde")).sel(
    altitude=slice(8000, 10000)
).plot(ax=axes[2], y="altitude", color="teal", label="n2 cutoff")
(rapsodi.ta.mean("sonde") - gate_new.ta.mean("sonde")).sel(
    altitude=slice(8000, 10000)
).plot(ax=axes[2], y="altitude", color="navy")
sns.despine()
axes[2].set_xlabel("$\Delta T$ / K")
axes[2].set_ylabel("altitude / m")
axes[2].legend()
fig.savefig("n2_qc_criteria.png", dpi=300, bbox_inches="tight")
