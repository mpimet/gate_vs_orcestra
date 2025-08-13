# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import moist_thermodynamics.functions as mt
from moist_thermodynamics import saturation_vapor_pressures as svp


import utilities.data_utils as data

from utilities.settings_and_colors import colors
import utilities.preprocessing as pp
import utilities.modify_ds as md

# %%


# %%
cids = data.get_cids()
datasets = {
    "rs": data.open_radiosondes(cids["radiosondes"]),
    "ds": data.open_dropsondes(cids["dropsondes"]),
    "gate": data.open_gate(cids["gate"]),
}

for name, ds in datasets.items():
    datasets[name] = (
        ds.pipe(pp.interpolate_gaps).pipe(pp.extrapolate_sfc).pipe(pp.sel_gate_A)
    )

# %%
datasets["rs"] = datasets["rs"].where(datasets["rs"].ascent_flag == 0, drop=True)
datasets["orcestra"] = xr.concat(
    [datasets["rs"], datasets["ds"]],
    dim="sonde",
)


# %%
es_liq = svp.liq_wagner_pruss
es_ice = svp.ice_wagner_etal
es_mixed = mt.make_es_mxd(es_liq=es_liq, es_ice=es_ice)


for name, ds in datasets.items():
    datasets[name] = ds.assign(
        cp_t=(
            ("sonde"),
            ds.ta.where((ds.altitude > 14000) & (ds.ta.count(dim="altitude") > 1900))
            .min(dim="altitude")
            .values,
            {"long_name": "cold point temperature", "units": "K"},
        ),
        cp_z=(
            ("sonde"),
            ds.ta.where((ds.altitude > 14000) & (ds.ta.count(dim="altitude") > 1900))
            .idxmin("altitude", fill_value=np.nan)
            .values,
            {"long_name": "cold point altitude", "units": "m"},
        ),
    )


for name, ds in datasets.items():
    datasets[name] = ds.assign(
        diff_to_cp=(ds.ta - ds.cp_t).where(ds.altitude > ds.cp_z),
        rh_ice=mt.specific_humidity_to_relative_humidity(
            ds.q,
            p=ds.p,
            T=ds.ta,
            es=es_ice,
        ),
    )

# %%
ta_datasets = {}
for name, ds in datasets.items():
    ta_datasets[name] = xr.merge(
        [
            md.get_hist_of_ta(
                ds.ta.sel(altitude=slice(0, 15000)),
                ds.rh.sel(altitude=slice(0, 15000)),
                var_binrange=(0, 1.1),
                ta_binrange=(220, 305),
            ).rename("rh_w"),
            md.get_hist_of_ta(
                ds.ta.sel(altitude=slice(0, 15000)),
                ds.rh_ice.sel(altitude=slice(0, 15000)),
                var_binrange=(0, 1.1),
                ta_binrange=(220, 305),
            ).rename("rh_ice"),
            md.get_hist_of_ta(
                ds.diff_to_cp,
                ds.rh_ice,
                var_binrange=(0, 1.1),
                ta_binrange=(0, 40),
            ).rename("rh_diff_cp"),
        ]
    )


# %%

# %%
thres = 1
distribution_t = [
    ("ta", "rh_w", 280),
    ("ta", "rh_ice", 255),
    ("diff_to_cp", "rh_diff_cp", 5),
]
labels = {
    "ta": "temperature",
    "diff_to_cp": "temperature difference to cold point",
    "rh_w": r"RH$_{\text{liq}}$",
    "rh_ice": r"RH$_{\text{ice}}$",
    "rh_diff_cp": r"RH$_{\text{ice}}$",
}
plt.style.use("utilities/gate.mplstyle")
fig, axes = plt.subplots(
    ncols=len(distribution_t), figsize=((len(distribution_t)) * 5, 5)
)
for name, cname in [("gate", "gate"), ("ds", "beach"), ("rs", "rapsodi")]:
    print(cname, colors[cname])
    for i, (ta_var, rh_var, height) in enumerate(distribution_t):
        if np.any(
            ~np.isnan(
                ta_datasets[name][rh_var].sel(
                    {ta_var: slice(height - thres, height + thres)}
                )
            )
        ):
            sns.histplot(
                data=np.ravel(
                    ta_datasets[name][rh_var]
                    .sel({ta_var: slice(height - thres, height + thres)})
                    .values
                ),
                kde=True,
                stat="density",
                element="step",
                binrange=(0, 1.1),
                ax=axes[i],
                binwidth=0.04,
                label=f"{cname.upper()}, {labels[rh_var]}",
                color=colors[cname],
            )
        else:
            continue

        axes[i].set_title(
            f"{labels[ta_var]} \n at {height}" + r" $\pm$ {thres} K".format(thres=thres)
        )

for ax in axes[1:]:
    ax.legend(loc="upper right")
axes[0].legend(loc="upper left")
for ax in axes:
    ax.set_xlabel("Relative Humidity")
    ax.set_xlim(0, 1.1)
sns.despine()
