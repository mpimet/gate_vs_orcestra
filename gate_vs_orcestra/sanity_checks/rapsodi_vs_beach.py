# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

import sys

sys.path.append("..")
import data_utils as data

# %%
cids = data.get_cids()
rs = data.open_radiosondes(cids["radiosondes"])
ds = data.open_dropsondes(cids["dropsondes"])
# %%
rapsodi_region = (
    data.get_gate_region(rs)
    .where(rs.ascent_flag == 0, drop=True)
    .drop_vars(["launch_time", "bin_average_time"])
    .interpolate_na(dim="altitude", method="linear", max_gap=1000)
    .interpolate_na(
        dim="altitude", method="linear", fill_value="extrapolate", max_gap=300
    )
)
beach_region = (
    data.get_gate_region(ds)
    .drop_vars(["launch_time", "bin_average_time"])
    .interpolate_na(dim="altitude", method="linear", max_gap=1000)
    .interpolate_na(
        dim="altitude", method="linear", fill_value="extrapolate", max_gap=300
    )
)
# %%
rapsodi_list = []
beach_list = []
for var, var_range in zip(
    ["theta", "ta", "rh"],
    [np.linspace(295, 360, 200), np.linspace(190, 305, 200), np.linspace(0, 1, 100)],
):
    rapsodi_list.append(
        data.get_hist_of_ta(
            np.log(rapsodi_region.p).sel(altitude=slice(0, 14000)),
            rapsodi_region[var].sel(altitude=slice(0, 14000)),
            bins_var=var_range,
            bins_ta=np.linspace(7, 12, 200),
        ).rename(var)
    )
    beach_list.append(
        data.get_hist_of_ta(
            np.log(beach_region.p).sel(altitude=slice(0, 14000)),
            beach_region[var].sel(altitude=slice(0, 14000)),
            bins_var=var_range,
            bins_ta=np.linspace(7, 12, 200),
        ).rename(var)
    )
rapsodi_p = xr.merge(rapsodi_list)
beach_p = xr.merge(beach_list)
# %%
rapsodi_p = rapsodi_p.assign_coords(p=np.exp(rapsodi_p.p))
beach_p = beach_p.assign_coords(p=np.exp(beach_p.p))
# %%
fig, ax = plt.subplots()
(rapsodi_p.theta.mean("sonde_id") - beach_p.theta.mean("sonde_id")).plot(
    y="p", ax=ax, label=r"$\Delta \theta$"
)
(rapsodi_p.ta.mean("sonde_id") - beach_p.ta.mean("sonde_id")).plot(
    y="p", ax=ax, label=r"$\Delta T$"
)
ax.invert_yaxis()
ax.legend()
ax.set_xlabel("Temperature difference / K")
sns.despine()
# %%
(rapsodi_region.p.mean("sonde_id") - beach_region.p.mean("sonde_id")).plot(y="altitude")

# %%
fig, ax = plt.subplots()
(
    (rapsodi_region.theta.mean("sonde_id") - beach_region.theta.mean("sonde_id"))
    - (rapsodi_region.ta.mean("sonde_id") - beach_region.ta.mean("sonde_id"))
).plot(y="altitude", ax=ax, label=r"$\Delta \theta$")
