# %%
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import moist_thermodynamics.functions as mt
from moist_thermodynamics import saturation_vapor_pressures as svp

import utilities.data_utils as data
import utilities.preprocessing as pp
import utilities.modify_ds as md
from utilities.settings_and_colors import colors  # noqa

# %%
cids = data.get_cids()
datasets = {
    "rs": data.open_radiosondes(cids["radiosondes"]),
    "ds": data.open_dropsondes(cids["dropsondes"]),
    "gate": data.open_gate(cids["gate"]),
}

for name, ds in datasets.items():
    datasets[name] = (
        ds.pipe(pp.interpolate_gaps).pipe(pp.extrapolate_sfc).pipe(pp.sel_percusion_E)
    )
# %%
datasets["rs"] = datasets["rs"].where(datasets["rs"].ascent_flag == 0, drop=True)
datasets["orcestra"] = xr.concat(
    [datasets["rs"], datasets["ds"]],
    dim="sonde",
)
# %%
ta_datasets = {name: [] for name in datasets.keys()}
for name, ds in datasets.items():
    ta_datasets[name].append(
        md.get_hist_of_ta(
            ds.ta.sel(altitude=slice(0, 14000)),
            ds.rh.sel(altitude=slice(0, 14000)),
            var_binrange=(0, 1.1),
            ta_binrange=(220, 305),
        ).rename("rh")
    )
for name, ds in datasets.items():
    ta_datasets[name].append(
        md.get_hist_of_ta(
            ds.ta.sel(altitude=slice(0, 14000)),
            ds.p.sel(altitude=slice(0, 14000)),
            var_binrange=(10000, 100000),
            ta_binrange=(220, 305),
        ).rename("p")
    )
for name, ds in datasets.items():
    ta_datasets[name] = xr.merge(ta_datasets[name])
# %%
ta_2d_datasets = {}
for name, ds in datasets.items():
    ta_2d_datasets[name] = md.get_hist_of_ta_2d(
        ds.ta.sel(altitude=slice(0, 14000)),
        ds.rh.sel(altitude=slice(0, 14000)),
        var_binrange=(0, 1.1),
        ta_binrange=(220, 305),
    )

# %%


es_liq = svp.liq_wagner_pruss
es_ice = svp.ice_wagner_etal
es_mixed = mt.make_es_mxd(es_liq=es_liq, es_ice=es_ice)


ice = mt.relative_humidity_to_specific_humidity(
    1.0,
    p=datasets["orcestra"].p.mean(dim="sonde"),
    T=datasets["orcestra"].ta.mean(dim="sonde"),
    es=es_ice,
)
ice_line = mt.specific_humidity_to_relative_humidity(
    ice,
    p=datasets["orcestra"].p.mean(dim="sonde"),
    T=datasets["orcestra"].ta.mean(dim="sonde"),
    es=es_liq,
)

rh = 0.95

p = 100000
T = 300
q_low = mt.relative_humidity_to_specific_humidity(
    rh,
    p=p,
    T=T,
    es=es_liq,
)
print("low q", q_low)
fix_q_rh_low = mt.specific_humidity_to_relative_humidity(
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
q_high = mt.relative_humidity_to_specific_humidity(
    rh,
    p=p,
    T=T,
    es=es_liq,
)
print("high q ", q_high)
fix_q_rh_high = mt.specific_humidity_to_relative_humidity(
    q_high,
    p=ta_datasets["orcestra"].p.mean(dim="sonde"),
    T=ta_datasets["orcestra"].ta,
    es=es_mixed,
)

# %%
cw = 190 / 25.4
sns.set_context("paper")
cs_threshold = 0.95
fig, axes = plt.subplots(ncols=2, figsize=(cw, cw / 2))
for ax in axes:
    ta_datasets["orcestra"].mean("sonde").rolling(ta=5).mean().rh.plot(
        label="ORCESTRA",
        y="ta",
        color="blue",
        ax=ax,
    )
    ta_datasets["gate"].mean("sonde").rolling(ta=5).mean().rh.plot(
        label="GATE",
        y="ta",
        color="red",
        ax=ax,
    )

for idx, (name, cmap) in enumerate([("gate", "Reds"), ("orcestra", "Blues")]):
    (ta_2d_datasets[name] / ta_2d_datasets[name].sum("rh_bin")).plot(
        vmax=0.03,
        cmap=cmap,
        ax=axes[idx],
        y="ta_bin",
        x="rh_bin",
        add_colorbar=False,
    )


bbox_args = dict(boxstyle="round", fc="white", alpha=0.3)
for ax in axes:
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.set_xlabel("Relative humidity")
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
axes[0].legend(loc=3)
axes[0].set_ylabel("Temperature / K")
axes[0].set_title("GATE")
axes[1].set_title("ORCESTRA")


fig.suptitle(
    "RH in the ORCESTRA-East subdomain",
)

sns.despine(offset={"bottom": 10})
fig.savefig(
    "images/rh_histograms.pdf",
    bbox_inches="tight",
)
# %%

# %%
pltcolors = sns.color_palette("Paired", n_colors=8)
cs_threshold = 0.98
fig, ax = plt.subplots(figsize=(5, 5))

for name, color_idx in [("orcestra", 0), ("gate", 4)]:
    ds = ta_datasets[name].rh
    ds.where(ds.max(dim="ta") < cs_threshold).mean("sonde").rolling(ta=5).mean().plot(
        y="ta",
        ax=ax,
        label=f"{name} rh_max < {cs_threshold:.2f}",
        c=pltcolors[color_idx],
    )
    ds.mean("sonde").rolling(ta=5).mean().plot(
        label=name, y="ta", ax=ax, c=pltcolors[color_idx + 1]
    )


ax.invert_yaxis()
ax.legend(loc="upper right", fontsize=12)
ax.set_xlabel("Relative humidity")
ax.set_ylabel("Temperature / K")
sns.despine(offset=10)
fig.savefig(
    "images/total_vs_clear_sky.pdf",
)
