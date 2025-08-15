# %%

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import numpy as np
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.constants as mtc
from moist_thermodynamics import saturation_vapor_pressures as svp

import utilities.data_utils as data
import utilities.preprocessing as pp
import utilities.mt_beta as mt_beta
from utilities.settings_and_colors import colors

# %%

cids = data.get_cids()
datasets = {
    "rapsodi": data.open_radiosondes(cids["radiosondes"]),
    "beach": data.open_dropsondes(cids["dropsondes"]),
    "gate": data.open_gate(cids["gate"]),
}
datasets["orcestra"] = xr.concat(
    [datasets["rapsodi"], datasets["beach"]],
    dim="sonde",
)

for name, ds in datasets.items():
    datasets[name] = (
        ds.pipe(pp.interpolate_gaps).pipe(pp.extrapolate_sfc).pipe(pp.sel_gate_A)
    )

# %%
for name, ds in datasets.items():
    datasets[name] = ds.where(
        np.all(~np.isnan(ds.sel(altitude=slice(0, 8000)).q), axis=1)
        & (ds.q > 0)
        & np.all(~np.isnan(ds.sel(altitude=slice(0, 8000)).p), axis=1)
        & np.all(~np.isnan(ds.sel(altitude=slice(0, 8000)).ta), axis=1),
    )


# %%
def density_from_q(p, T, q):
    Rd = mtc.dry_air_gas_constant
    Rv = mtc.water_vapor_gas_constant
    return p / ((Rd + (Rv - Rd) * q) * T)


def calc_iwv(ds):
    ds = ds.assign(rho=density_from_q(ds.p, ds.ta, ds.q))
    iwv = (ds.q * ds.rho).fillna(0).integrate(coord="altitude")
    ds = ds.assign(iwv=xr.where(iwv > 0, iwv, np.nan))
    return ds


# %%
for name, ds in datasets.items():
    datasets[name] = calc_iwv(ds)
# %%


# %%


P = np.arange(100900.0, 4000.0, -500)
orc_q = (
    datasets["orcestra"]
    .mean("sonde")
    .swap_dims({"altitude": "p"})
    .dropna(dim="p", how="any", subset=["p"])
    .interp(p=P)
    .q
)
gate_q = (
    datasets["gate"]
    .mean("sonde")
    .swap_dims({"altitude": "p"})
    .dropna(dim="p", how="any", subset=["p"])
    .interp(p=P)
    .q
)
orc_sfc = datasets["orcestra"].sel(altitude=0).mean("sonde")
gate_sfc = datasets["gate"].sel(altitude=0).mean("sonde")
orc_pseudo = calc_iwv(
    mt_beta.mk_sounding_ds(
        P, orc_sfc.ta.values, orc_q.values, thx=mtf.theta_e_bolton
    ).rename({"T": "ta", "P": "p"})
)
gate_pseudo = calc_iwv(
    mt_beta.mk_sounding_ds(
        P, gate_sfc.ta.values, gate_q.values, thx=mtf.theta_e_bolton
    ).rename({"T": "ta", "P": "p"})
)
# %%
gate_orcrh_q = mtf.relative_humidity_to_specific_humidity(
    RH=mtf.specific_humidity_to_relative_humidity(
        orc_q.values, P, orc_pseudo.ta.values, es=svp.liq_wagner_pruss
    ),
    p=P,
    T=gate_pseudo.ta.values,
)
gate_orcq_pseudo = calc_iwv(
    mt_beta.mk_sounding_ds(
        P, gate_sfc.ta.values, gate_orcrh_q, thx=mtf.theta_e_bolton
    ).rename({"T": "ta", "P": "p"})
)

# %%
plt.style.use("utilities/gate.mplstyle")
fig, ax = plt.subplots(figsize=(6, 5.5))
for name in ["rapsodi", "beach", "gate"]:
    sns.histplot(
        data=datasets[name].iwv,
        bins=30,
        element="step",
        stat="density",
        label=name,
        color=colors[name],
    )

ax.axvline(x=datasets["orcestra"].iwv.mean(), color="#6d88bc", linestyle="-", alpha=0.5)
ax.text(
    x=datasets["orcestra"].iwv.mean(),
    y=0.2,
    s="Orcestra",
    color="#6d88bc",
    ha="right",
    va="top",
    rotation=90,
)

ax.axvline(
    x=datasets["gate"].iwv.mean(), color=colors["gate"], linestyle="-", alpha=0.5
)
ax.text(
    x=datasets["gate"].iwv.mean(),
    y=0.2,
    s="Gate",
    color=colors["gate"],
    ha="right",
    va="top",
    rotation=90,
)

ax.axvline(gate_orcq_pseudo.iwv, color="k", linestyle="--")
ax.text(
    x=gate_orcq_pseudo.iwv + 0.3,
    y=0.2,
    s="Gate Orcestra RH",
    color="k",
    ha="left",
    va="top",
    rotation=90,
)


ax.legend()
sns.despine(offset={"left": 5})
