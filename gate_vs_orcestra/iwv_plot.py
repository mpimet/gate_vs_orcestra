# %%

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import numpy as np
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.constants as mtc

import utilities.data_utils as data
import utilities.preprocessing as pp

import utilities.thermo as thermo

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
        ds.pipe(pp.interpolate_gaps).pipe(pp.extrapolate_sfc).pipe(pp.sel_percusion_E)
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
P = np.arange(100900.0, 4000.0, -500)

orc_sfc = datasets["orcestra"].sel(altitude=0).mean("sonde")
gate_sfc = datasets["gate"].sel(altitude=0).mean("sonde")

orc_pseudo = thermo.make_sounding_from_adiabat(
    P, orc_sfc.ta.values, orc_sfc.q.values, thx=mtf.theta_e_bolton
).rename({"T": "ta", "P": "p"})
orc_pseudo = orc_pseudo.where(orc_pseudo.ta > 200, drop=True)
gate_pseudo = thermo.make_sounding_from_adiabat(
    P, gate_sfc.ta.values, gate_sfc.q.values, thx=mtf.theta_e_bolton
).rename({"T": "ta", "P": "p"})
gate_pseudo = gate_pseudo.where(gate_pseudo.ta > 200, drop=True)


# %%
def get_rh(datasets, rhname, pseudo_ds):
    mean_ta = datasets[rhname].mean(dim="sonde").dropna(dim="altitude", subset=["ta"])
    _, idx = np.unique(mean_ta.ta, return_index=True)
    lcl = (
        pseudo_ds.swap_dims({"altitude": "p"})
        .sel(
            p=mtf.plcl_bolton(
                mean_ta.ta.sel(altitude=0),
                mean_ta.p.sel(altitude=0),
                mean_ta.q.sel(altitude=0),
            ),
            method="nearest",
        )
        .altitude
    )
    return xr.concat(
        [
            mean_ta.rh.interp(
                altitude=pseudo_ds.altitude.sel(altitude=slice(None, lcl))
            ).isel(altitude=slice(0, -1)),
            (
                mean_ta.isel(altitude=idx)
                .swap_dims({"altitude": "ta"})
                .rh.interp(ta=pseudo_ds.ta.values)
                .to_dataset()
                .assign(
                    altitude=("ta", pseudo_ds.altitude.values),
                )
                .swap_dims({"ta": "altitude"})
                .sel(altitude=slice(lcl, None))
                .reset_coords("ta")
                .rh
            ),
        ],
        dim="altitude",
    )


ref_rh = "orcestra"
gate_pseudo = gate_pseudo.assign(
    rh=("altitude", get_rh(datasets, ref_rh, gate_pseudo).values)
).sel(altitude=slice(0, 11500))
orc_pseudo = orc_pseudo.assign(
    rh=("altitude", get_rh(datasets, ref_rh, orc_pseudo).values)
).sel(altitude=slice(0, 11500))
# %%

# %%
gate_pseudo = calc_iwv(
    gate_pseudo.assign(
        q=mtf.relative_humidity_to_specific_humidity(
            RH=gate_pseudo.rh + 0.015,
            p=gate_pseudo.p,
            T=gate_pseudo.ta,
        )
    )
)
orc_pseudo = calc_iwv(
    orc_pseudo.assign(
        q=mtf.relative_humidity_to_specific_humidity(
            RH=orc_pseudo.rh,
            p=orc_pseudo.p,
            T=orc_pseudo.ta,
        )
    )
)
# %%
cw = 190 / 25.4
sns.set_context("paper")
fig, ax = plt.subplots(figsize=(cw / 2, cw / 2 * 0.75))
for name in ["rapsodi", "beach", "gate"]:
    sns.histplot(
        data=datasets[name].iwv,
        bins=30,
        element="step",
        stat="density",
        label=name,
        color=colors[name],
    )

print("orcestra median", datasets["orcestra"].iwv.median().values)
print("gate median", datasets["gate"].iwv.median().values)

ax.axvline(
    x=datasets["orcestra"].iwv.median(),
    ymax=0.91,
    color="#6d88bc",
    linestyle="-",
    alpha=0.5,
)
ax.text(
    x=datasets["orcestra"].iwv.median(),
    y=0.2,
    fontsize=8,
    s="Orcestra",
    color="#6d88bc",
    ha="left",
    va="top",
    rotation=90,
)

ax.axvline(
    x=datasets["gate"].iwv.median(),
    ymax=0.91,
    color=colors["gate"],
    linestyle="-",
    alpha=0.5,
)
ax.text(
    x=datasets["gate"].iwv.median(),
    y=0.2,
    fontsize=8,
    s="GATE",
    color=colors["gate"],
    ha="right",
    va="top",
    rotation=90,
)

ax.axvline(orc_pseudo.iwv, ymax=0.9, color="#6d88bc", linestyle="--")
# ax.text(
#     x=orc_pseudo.iwv + 0.3,
#     y=0.2,
#     fontsize=8,
#     s="pseudo",
#     color="#6d88bc",
#     ha="left",
#     va="top",
#     rotation=90,
# )
ax.axvline(gate_pseudo.iwv, ymax=0.9, color=colors["gate"], linestyle="--")
# ax.text(
#     x=gate_pseudo.iwv + 0.2,
#     y=0.2,
#     fontsize=8,
#     s="pseudo",
#     color=colors["gate"],
#     ha="left",
#     va="top",
#     rotation=90,
# )

print("orcestra pseudo", orc_pseudo.iwv.values)
print("gate pseudo", gate_pseudo.iwv.values)

mean_pseudo = (orc_pseudo.iwv + gate_pseudo.iwv).values / 2
diff_pseudo = (orc_pseudo.iwv - gate_pseudo.iwv).values

mean_campaigns = (
    datasets["orcestra"].iwv.median() + datasets["gate"].iwv.median()
).values / 2
diff_campaigns = datasets["orcestra"].iwv.median() - datasets["gate"].iwv.median()

ax.annotate(
    "{:.2f}".format(diff_pseudo),
    xy=(mean_pseudo, 0.205),
    xytext=(mean_pseudo, 0.21),
    fontsize=8,
    ha="center",
    va="bottom",
    arrowprops=dict(arrowstyle="-[, widthB=2, lengthB=.1", lw=2.0),
)

ax.annotate(
    "{:.2f}".format(diff_campaigns),
    xy=(mean_campaigns, 0.21),
    xytext=(mean_campaigns, 0.215),
    fontsize=8,
    ha="center",
    va="bottom",
    alpha=0.5,
    arrowprops=dict(arrowstyle="-[, widthB=2, lengthB=.1", lw=2.0, color="gray"),
)

ax.set_ylim(0, 0.218)
ax.legend(fontsize=8)
ax.set_yticks(ticks=np.arange(0, 0.2, 0.025))
ax.spines["left"].set_bounds(-0.0, 0.19)
sns.despine(offset={"left": 5})
ax.set_xlabel("IWV / kg m$^{-2}$")
fig.savefig("plots/iwv_histograms.pdf", bbox_inches="tight")
# %%
