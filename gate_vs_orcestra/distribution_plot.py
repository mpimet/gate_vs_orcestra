# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import moist_thermodynamics.functions as mt
from moist_thermodynamics import saturation_vapor_pressures as svp

from moist_thermodynamics import constants
import data_utils as data
from settings_and_colors import colors
# %%


cids = data.get_cids()
rs = data.open_radiosondes(cids["radiosondes"])
ds = data.open_dropsondes(cids["dropsondes"])

gate = data.open_gate(cids["gate"])

# unique, keep = np.unique(gate.sonde_id.values, return_index=True)
# gate = gate.isel(sonde_id=keep)
# %%

# %%
gate_region = (
    data.get_gate_region(gate)
    .drop_vars("launch_time")
    .interpolate_na(dim="altitude", method="linear", max_gap=1000)
    .interpolate_na(
        dim="altitude", method="linear", fill_value="extrapolate", max_gap=300
    )
)
# %%
orcestra_gate = (
    xr.concat(
        [
            data.get_gate_region(rs).where(rs.ascent_flag == 0, drop=True),
            data.get_gate_region(ds),
        ],
        dim="sonde_id",
    )
    .drop_vars(["launch_time", "bin_average_time"])
    .interpolate_na(dim="altitude", method="linear", max_gap=1000)
    .interpolate_na(
        dim="altitude", method="linear", fill_value="extrapolate", max_gap=300
    )
)


# %%
es_liq = svp.liq_analytic
es_ice = svp.ice_analytic
es = mt.make_es_mxd(es_liq=svp.liq_analytic, es_ice=svp.ice_analytic)
Rd = constants.Rd
Rv = constants.Rv
cpd = constants.cpd
P0 = constants.P0


def get_rh(T, q, p, es):
    x = es(T) * Rd / Rv / (p - es(T))
    return q * (1 + x) / x


def get_q(T, rh, p, es):
    x = es(T) * Rd / Rv / (p - es(T))
    return rh * x / (1 + x)


q = get_q(195, 1, 150, es=es_ice)
rh_5 = get_rh(200, q, 150, es=es_ice)
# %%

gate_region = gate_region.assign(
    rh_ice=get_rh(gate_region.ta, gate_region.q, gate_region.p, es=svp.ice_analytic),
    rh_liq=get_rh(gate_region.ta, gate_region.q, gate_region.p, es=svp.liq_analytic),
)
orcestra_gate = orcestra_gate.assign(
    rh_ice=get_rh(
        orcestra_gate.ta, orcestra_gate.q, orcestra_gate.p, es=svp.ice_analytic
    ),
    rh_liq=get_rh(
        orcestra_gate.ta, orcestra_gate.q, orcestra_gate.p, es=svp.liq_analytic
    ),
)

# %%
orcestra_cp = orcestra_gate.ta.where(orcestra_gate.altitude > 15000).min(dim="altitude")
gate_cp = gate_region.ta.where(
    (gate_region.ta.count(dim="altitude") > 1900) & (gate_region.altitude > 15000)
).min(dim="altitude")

# %%
gate_strato = gate_region.where(
    (gate_region.altitude > gate_region.ta.argmin(dim="altitude") * 10)
    & (gate_region.ta.count(dim="altitude") > 1900)
    & (gate_region.altitude > 14000)
)
orcestra_strato = orcestra_gate.where(
    (
        orcestra_gate.altitude
        > orcestra_gate.ta.dropna(dim="sonde_id", how="all").argmin(dim="altitude") * 10
    )
    & (orcestra_gate.altitude > 14000)
)
# %%rs_strato = rs_strato.assign(diff_to_cp=rs_strato.ta - rs_strato.ta.min(dim="altitude"))
gate_strato = gate_strato.assign(
    diff_to_cp=gate_strato.ta - gate_strato.ta.min(dim="altitude")
)
orcestra_strato = orcestra_strato.assign(
    diff_to_cp=orcestra_strato.ta - orcestra_strato.ta.min(dim="altitude")
)

# %%
orcestra_ta = data.get_hist_of_ta(
    orcestra_gate.ta.sel(altitude=slice(0, 14000)),
    orcestra_gate.rh.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)

gate_ta = data.get_hist_of_ta(
    gate_region.ta.sel(altitude=slice(0, 14000)),
    gate_region.rh.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)
orcestra_ta_ice = data.get_hist_of_ta(
    orcestra_gate.ta.sel(altitude=slice(0, 14000)),
    orcestra_gate.rh_ice.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)

gate_ta_ice = data.get_hist_of_ta(
    gate_region.ta.sel(altitude=slice(0, 14000)),
    gate_region.rh_ice.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)


# %%
distribution_t = [280, 255]
thres = 5
plt.style.use("./gate.mplstyle")
fig, axes = plt.subplots(
    ncols=len(distribution_t) + 1, figsize=((len(distribution_t) + 1) * 5, 5)
)
ax = axes[-1]
for ax, t in zip(axes[:-1], distribution_t):
    if t > 273.15:
        rs_ds = orcestra_ta
        gate_ds = gate_ta
        subscript = "liq"
    else:
        rs_ds = orcestra_ta_ice
        gate_ds = gate_ta_ice
        subscript = "ice"
    sns.histplot(
        rs_ds.sel(ta=t, method="nearest"),
        bins=30,
        binrange=(0, 1.1),
        stat="density",
        label=r"ORCESTRA (RH$_{{{}}}$)".format(subscript),
        kde=True,
        element="step",
        color=colors["orcestra"],
        ax=ax,
    )
    sns.histplot(
        gate_ds.sel(ta=t, method="nearest"),
        bins=30,
        binrange=(0, 1.1),
        stat="density",
        label=r"GATE (RH$_{{{}}}$)".format(subscript),
        kde=True,
        element="step",
        color=colors["gate"],
        ax=ax,
    )
    ax.set_title(f"troposphere temperature: \n {t} K")

ax = axes[-1]
sns.histplot(
    orcestra_strato.rh_ice.where(
        (orcestra_strato.diff_to_cp > thres)
        & (orcestra_strato.diff_to_cp < thres + 0.5)
    ).mean(dim="altitude"),
    bins=30,
    stat="density",
    label=r"RAPSODI (RH$_{{ice}}$)",
    kde=True,
    binrange=(0, 1.1),
    color=colors["rapsodi"],
    element="step",
    ax=ax,
)
sns.histplot(
    gate_strato.rh_ice.where(
        (gate_strato.diff_to_cp > thres) & (gate_strato.diff_to_cp < thres + 0.5)
    ).mean(dim="altitude"),
    bins=30,
    binrange=(0, 1.1),
    stat="density",
    label=r"GATE (RH$_{{ice}}$)",
    kde=True,
    element="step",
    color=colors["gate"],
    ax=ax,
)

ax.set_title(f"stratosphere: \n {thres} K < $T$ - $T_{{min}}$ < {thres + 0.5} K")


for ax in axes:
    ax.legend(loc="upper right")
    ax.set_xlabel("Relative Humidity")
    ax.set_xlim(0, 1.1)
sns.despine()
