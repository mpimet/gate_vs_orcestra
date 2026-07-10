# %%
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import numpy as np
import moist_thermodynamics.functions as mtf
import utilities.thermo as thermo
from moist_thermodynamics import saturation_vapor_pressures as svp
import moist_thermodynamics.constants as mtc
import utilities.data_utils as dus
import utilities.preprocessing as pp
import utilities.modify_ds as md
import utilities.settings_and_colors as set

es = svp.liq_wagner_pruss
# %%
cids = dus.get_cids()
datasets = {
    "rapsodi": dus.open_radiosondes(cids["radiosondes"]),
    "beach": dus.open_dropsondes(cids["dropsondes"]),
    "gate": dus.open_gate(cids["gate"]),
}

for name, ds in datasets.items():
    datasets[name] = (
        ds.pipe(pp.interpolate_gaps)
        .pipe(pp.extrapolate_sfc)
        .pipe(pp.sel_percusion_E)
        .pipe(pp.sel_itcz)
    )
# %%
datasets["orcestra"] = xr.concat(
    [datasets["rapsodi"], datasets["beach"]],
    dim="sonde",
)


# %% RH-T histograms


ta_bin_num = 200
var_bin_num = 100
ta_datasets = {name: [] for name in datasets.keys()}
for name, ds in datasets.items():
    ta_datasets[name] = xr.merge(
        [
            md.get_hist_of_ta(
                ds.sel(altitude=slice(0, 14000)),
                ta_binrange=(220, 305),
                ta_bin_num=ta_bin_num,
                var=var,
            )
            for var in ["p", "rh"]
        ]
    )
# %%
ta_2d_datasets = {}
for name, ds in datasets.items():
    ta_2d_datasets[name] = md.get_hist_of_ta_2d(
        ds.ta.sel(altitude=slice(0, 14000)),
        ds.rh.sel(altitude=slice(0, 14000)),
        var_binrange=(0, 1.1),
        ta_binrange=(220, 305),
        var_bin_num=var_bin_num,
        ta_bin_num=ta_bin_num,
    )

# %% additional lines for RH-T plot


es_liq = es
es_ice = svp.ice_wagner_etal
es_mixed = mtf.make_es_mxd(es_liq=es_liq, es_ice=es_ice)


ice = mtf.relative_humidity_to_specific_humidity(
    1.0,
    p=datasets["orcestra"].p.mean(dim="sonde"),
    T=datasets["orcestra"].ta.mean(dim="sonde"),
    es=es_ice,
)
ice_line = mtf.specific_humidity_to_relative_humidity(
    ice,
    p=datasets["orcestra"].p.mean(dim="sonde"),
    T=datasets["orcestra"].ta.mean(dim="sonde"),
    es=es_liq,
)

rh = 0.95

p = 100000
T = 300
q_low = mtf.relative_humidity_to_specific_humidity(
    rh,
    p=p,
    T=T,
    es=es_liq,
)
print("low q", q_low)
fix_q_rh_low = mtf.specific_humidity_to_relative_humidity(
    q_low,
    p=ta_datasets["orcestra"]
    .p.mean(dim="sonde")
    .interpolate_na(dim="ta", fill_value="extrapolate", method="linear"),
    T=ta_datasets["orcestra"].ta,
    es=es_liq,
)

T = 230
p = 23000
rh = 0.4
q_high = mtf.relative_humidity_to_specific_humidity(
    rh,
    p=p,
    T=T,
    es=es_liq,
)
print("high q ", q_high)
fix_q_rh_high = mtf.specific_humidity_to_relative_humidity(
    q_high,
    p=ta_datasets["orcestra"].p.mean(dim="sonde"),
    T=ta_datasets["orcestra"].ta,
    es=es_mixed,
)
# %% get iwv

iwv = {}
for name, ds in datasets.items():
    iwv[name] = ds.where(
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


for name, ds in iwv.items():
    iwv[name] = calc_iwv(ds)

# %%

sfc_est = set.sfc_est
Px = set.Px
P = set.P
for key in sfc_est.keys():
    sfc_est[key]["q"] = mtf.partial_pressure_to_specific_humidity(
        svp.es_default(sfc_est[key]["T"]) * sfc_est[key]["RH"], Px
    )

adiabat_fits = {
    key: thermo.make_sounding_from_adiabat(
        P, sfc_est[key]["T"], sfc_est[key]["q"]
    ).rename({"T": "ta", "P": "p"})
    for key in sfc_est.keys()
}


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
adiabat_fits = {
    key: ds.assign(rh=("altitude", get_rh(iwv, ref_rh, ds).values)).sel(
        altitude=slice(0, 11500)
    )
    for key, ds in adiabat_fits.items()
}

# %%

fig, ax = plt.subplots()
for key, ds in adiabat_fits.items():
    ax.plot(ds.rh, ds.ta, label=key, color=set.colors[key])

ax.invert_yaxis()
sns.despine()
# %%

adiabat_fits = {
    key: calc_iwv(
        ds.assign(
            q=mtf.relative_humidity_to_specific_humidity(
                RH=ds.rh, p=ds.p, T=ds.ta, es=es
            )
        )
    )
    for key, ds in adiabat_fits.items()
}

# %%

# %%
cw = 190 / 25.4
sns.set_context("paper")

fig, axes = plt.subplots(ncols=2, figsize=(cw, cw / 2), width_ratios=[0.6, 0.45])

for name, s, offset, ha in [
    #    ("rapsodi", "ORCESTRA-RS"),
    #   ("beach", "ORCESTRA-DS"),
    ("gate", "GATE", 0.0, "right"),
    ("orcestra", "ORC", 0.5, "left"),
]:
    sns.histplot(
        data=iwv[name].iwv,
        bins=40,
        binrange=(32, 73),
        element="step",
        stat="density",
        label=name.upper(),
        color=set.colors[name],
        ax=axes[0],
    )
    print(f"{name} median", iwv[name].iwv.median().values)
    axes[0].axvline(
        x=iwv[name].iwv.median(),
        ymax=0.8,
        color=set.colors[name],
        linestyle="-",
        linewidth=2,
        alpha=0.5,
    )
    axes[0].text(
        x=iwv[name].iwv.median() + offset,
        y=0.12,
        fontsize=8,
        s=s,
        color=set.colors[name],
        ha=ha,
        va="top",
        rotation=90,
    )
    axes[0].axvline(
        adiabat_fits[name].iwv, ymax=0.8, color=set.colors[name], linestyle="--"
    )
    print(f"{name} pseudo", adiabat_fits[name].iwv.values)


for ds, y, alpha, c in [(adiabat_fits, 0.123, 1, "k"), (iwv, 0.134, 0.5, "gray")]:
    mean = (ds["orcestra"].iwv.median() + ds["gate"].iwv.median()).values / 2
    diff = (ds["orcestra"].iwv.median() - ds["gate"].iwv.median()).values

    axes[0].annotate(
        "{:.2f}".format(diff),
        xy=(mean, y),
        xytext=(mean, y + 0.005),
        fontsize=8,
        ha="center",
        alpha=alpha,
        va="bottom",
        arrowprops=dict(arrowstyle="-[, widthB=2.5, lengthB=.1", lw=2.0, color=c),
    )
axes[0].set_ylim(None, 0.15)
axes[0].set_xlabel("IWV / kg m$^{-2}$")

# RH
for name in ["orcestra", "gate"]:
    ta_datasets[name].mean("sonde").rh.plot(
        label=name.upper(),
        y="ta",
        color=set.colors[name],
        linewidth=2,
        ax=axes[1],
    )
    axes[1].fill_betweenx(
        ta_datasets[name].ta,
        ta_datasets[name].rh.quantile(0.1, dim="sonde"),
        ta_datasets[name].rh.quantile(0.9, dim="sonde"),
        alpha=0.1,
        color=set.colors[name],
    )
axes[1].invert_yaxis()
axes[1].legend(loc="upper right")
axes[1].set_ylabel("$T$ / K")
axes[1].set_xlabel("RH / 1")
axes[1].axhline(273.15, color="k", linestyle="--")
axes[1].plot(
    ice_line.values,
    datasets["orcestra"].ta.mean("sonde"),
    color="black",
)
bbox_args = dict(boxstyle="round", fc="white", alpha=0.3)
axes[1].annotate(
    r"RH$_{\text{ice}} = 1$",
    xy=(1.1, 250),
    xycoords="data",
    fontsize=7,
    ha="right",
    va="top",
    bbox=bbox_args,
)
axes[1].plot(
    fix_q_rh_low.values,
    ta_datasets["orcestra"].ta,
    color="black",
    linestyle="-",
)
axes[1].annotate(
    "q = {:.4f}".format(q_low),
    xy=(1.1, 303),
    xycoords="data",
    fontsize=7,
    ha="right",
    va="top",
    bbox=bbox_args,
)
axes[1].set_ylim(305, 220)
axes[1].set_yticks(
    [305, 295, 285, 273.15, 265, 255, 245, 235, 225],
    labels=["305", "295", "285", "273.15", "265", "255", "245", "235", "225"],
)
axes[1].set_xlim(0, 1.1)

sns.despine(ax=axes[1], offset={"bottom": 10})
sns.despine(ax=axes[0], offset=10)
fig.tight_layout()
fig.savefig("iwv_rh_6to11.pdf", bbox_inches="tight")
# %%
cw = 190 / 25.4
sns.set_context("paper")
cs_threshold = 0.95
gate_cmap = sns.light_palette(set.colors["gate"], as_cmap=True)
orc_cmap = sns.light_palette("cornflowerblue", as_cmap=True)

fig, axes = plt.subplot_mosaic(
    [["top", "top"], ["left", "right"]],
    figsize=(cw, cw * 0.75),
    height_ratios=[0.25, 0.8],
)

for ax in [axes["left"], axes["right"]]:
    ta_datasets["orcestra"].mean("sonde").rh.plot(
        label="ORCESTRA",
        y="ta",
        color=set.colors["orcestra"],
        linewidth=2,
        ax=ax,
    )
    ta_datasets["gate"].mean("sonde").rh.plot(
        label="GATE",
        y="ta",
        color=set.colors["gate"],
        linewidth=2,
        ax=ax,
    )

for idx, (name, cmap, pos) in enumerate(
    [("gate", gate_cmap, "left"), ("orcestra", orc_cmap, "right")]
):
    (ta_2d_datasets[name] / ta_2d_datasets[name].sum("rh_bin")).plot(
        vmax=0.03,
        cmap=cmap,
        ax=axes[pos],
        y="ta_bin",
        x="rh_bin",
        alpha=0.8,
        add_colorbar=False,
    )

bbox_args = dict(boxstyle="round", fc="white", alpha=0.3)
for ax in [axes["left"], axes["right"]]:
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.set_xlabel("RH / 1")
    ax.axhline(273.15, color="k", linestyle="--")
    ax.plot(
        ice_line.values,
        datasets["orcestra"].ta.mean("sonde"),
        color="black",
    )
    ax.annotate(
        r"RH$_{\text{ice}} = 1$",
        xy=(1.1, 250),
        xycoords="data",
        ha="right",
        va="top",
        bbox=bbox_args,
    )
    ax.plot(
        fix_q_rh_low.values,
        ta_datasets["orcestra"].ta,
        color="black",
        linestyle="--",
    )
    ax.annotate(
        "q = {:.4f}".format(q_low),
        xy=(1.1, 302),
        xycoords="data",
        ha="right",
        va="top",
        bbox=bbox_args,
    )
    ax.plot(
        fix_q_rh_high.values,
        ta_datasets["orcestra"].ta,
        color="black",
        linestyle=":",
    )
    ax.annotate(
        "q = {:.5f}".format(q_high),
        xy=(1.1, 220),
        xycoords="data",
        ha="right",
        va="top",
        bbox=bbox_args,
    )
axes["left"].legend(loc=3)
axes["left"].set_ylabel("$T$ / K")
axes["left"].set_xlim(0, 1.05)
axes["left"].set_ylim(None, 220)

axes["right"].set_yticklabels([])
axes["right"].set_xlim(0, 1.1)
axes["right"].set_ylim(None, 220)
ax = axes["top"]
for name, label in [
    ("rapsodi", "ORCESTRA-RS"),
    ("beach", "ORCESTRA-DS"),
    ("gate", "GATE"),
]:
    sns.histplot(
        data=iwv[name].iwv,
        bins=40,
        binrange=(32, 73),
        element="step",
        stat="density",
        label=label,
        color=set.colors[name],
        ax=axes["top"],
    )

print("orcestra median", iwv["orcestra"].iwv.median().values)
print("gate median", iwv["gate"].iwv.median().values)
ref = "orcestra"
ax.axvline(
    x=iwv[ref].iwv.median(),
    ymax=0.91,
    color=set.colors[ref],
    linestyle="-",
    linewidth=2,
    alpha=0.5,
)
ax.text(
    x=iwv[ref].iwv.median() + 0.2,
    y=0.2,
    fontsize=8,
    s="ORC",
    color=set.colors[ref],
    ha="left",
    va="top",
    rotation=90,
)

ax.axvline(
    x=iwv["gate"].iwv.median(),
    ymax=0.91,
    color=set.colors["gate"],
    linestyle="-",
    alpha=0.5,
)
ax.text(
    x=iwv["gate"].iwv.median(),
    y=0.2,
    fontsize=8,
    s="GATE",
    color=set.colors["gate"],
    ha="right",
    va="top",
    rotation=90,
)

ax.set_ylim(0, 0.218)
ax.legend(fontsize=8)
ax.set_yticks(ticks=np.arange(0, 0.2, 0.08))
ax.set_xlabel("IWV / kg m$^{-2}$")
ax.spines["left"].set_bounds(-0.0, 0.19)

ax.set_ylabel("prob. density")
ax.set_xticks(
    [
        40.0,
        np.round(iwv["gate"].iwv.median(), 1),
        np.round(iwv[ref].iwv.median(), 1),
    ]
)

sns.despine(offset={"bottom": 10})
fig.tight_layout()
# %%
fig.savefig(
    "plots/rh_histograms.pdf",
    bbox_inches="tight",
)


# %%
pltcolors = sns.color_palette("Paired", n_colors=8)
cs_threshold = 0.98
cw = 190 / 25.4
fig, ax = plt.subplots(figsize=(cw / 2, cw / 2))

for name in ["beach", "gate", "rapsodi", "orcestra"]:
    ds = ta_datasets[name].rh
    ds = ds.where(
        (datasets[name].launch_lat > 6) & (datasets[name].launch_lat < 11), drop=True
    )
    print(ds.sizes)
    ds.mean("sonde").rolling(ta=5).mean().plot(
        label=name, y="ta", ax=ax, c=set.colors[name]
    )
    # for name in ["orcestra", "gate"]:
    ds = ta_datasets[name].rh
    ax.fill_betweenx(
        ds.ta,
        ds.quantile(0.1, dim="sonde").rolling(ta=5).mean(),
        ds.quantile(0.9, dim="sonde").rolling(ta=5).mean(),
        alpha=0.1,
        color=set.colors[name],
    )


ax.invert_yaxis()
ax.legend(loc="upper right", fontsize=10)
ax.set_xlabel("RH / 1")
ax.set_ylabel("$T$ / K")
sns.despine(offset=10)
fig.savefig(
    "plots/rh_mean.pdf",
)
# %%
