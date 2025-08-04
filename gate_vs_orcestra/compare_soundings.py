#  %%
# -Define some functions for plotting and evaluating atmospheric soundings
#
import utilities.data_utils as dus
import utilities.preprocessing as dpp
from utilities.settings_and_colors import colors

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from moist_thermodynamics import functions as mt
from moist_thermodynamics import saturation_vapor_pressures as svp
from moist_thermodynamics import constants
import utilities.mt_beta as mt_beta

import seaborn as sns

es = svp.liq_wagner_pruss
P0 = constants.P0
Rd = constants.Rd
Rv = constants.Rv

kappa = constants.Rd / constants.cpd
# %%
halo = (
    xr.open_dataset(
        "ipfs://bafybeif52irmuurpb27cujwpqhtbg5w6maw4d7zppg2lqgpew25gs5eczm",
        engine="zarr",
    )
    .rename_vars(
        {
            "IRS_LAT": "latitude",
            "IRS_LON": "longitude",
            "IRS_ALT": "altitude",
        }
    )
    .set_coords(({"latitude", "longitude", "altitude"}))
).pipe(dpp.sel_percusion_E, item_var="TIME", lon_var="longitude", lat_var="latitude")

hal = (
    halo.swap_dims({"TIME": "altitude"})
    .sortby("altitude")
    .sel(altitude=slice(14300, None))
)
halo_rh = halo.RELHUM.where(halo.RELHUM < 70, drop=True) / 100.0
x = hal.swap_dims({"altitude": "TIME"})
zbar = x.altitude.mean().values
# %%
# - load data
#
cids = dus.get_cids()
beach = dus.open_dropsondes(cids["dropsondes"])
rapsodi = dus.open_radiosondes(cids["radiosondes"])
# gate = dus.open_gate(cids["gate"])

gate = (
    xr.open_dataset(f"~/data/gate-xx.zarr", engine="zarr")
    .set_coords(["launch_lat", "launch_lon", "launch_time"])
    .swap_dims({"sonde": "launch_time"})
    .sel(launch_time=slice("1974-08-10", "1974-09-30"))
    .swap_dims({"launch_time": "sonde"})
)


# %%
# - localize data into different domains
#
gs_PE = dpp.sel_percusion_E(gate)
rs_PE = dpp.sel_percusion_E(rapsodi)
bs_PE = dpp.sel_percusion_E(beach)

gs_GA = dpp.sel_gate_A(gs_PE)
rs_GA = dpp.sel_gate_A(rs_PE)
bs_GA = dpp.sel_gate_A(bs_PE)

gs_bar_PE = gs_PE.mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()
rs_bar_PE = rs_PE.mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()
bs_bar_PE = bs_PE.mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()

gs_GA_bar = gs_GA.mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()
rs_GA_bar = rs_GA.mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()
bs_GA_bar = bs_GA.mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()

# %%
# - parameters for soundings
#
sfc_vals = {}
sondes = {"gate": gs_GA, "rapsodi": rs_GA}
for sset in ["gate", "rapsodi"]:
    xs = sondes[sset]
    P_sfc = xs.p.sel(altitude=slice(10, 50)).max(dim="altitude") + 100
    RH_sfc = xs.rh.sel(altitude=slice(0, 30)).max(dim="altitude")
    T_sfc = (
        xs.theta.sel(altitude=slice(200, 600)).mean(dim="altitude")
        * (P_sfc / P0) ** kappa
    )
    q_sfc = mt.partial_pressure_to_specific_humidity(RH_sfc * es(T_sfc), P_sfc)
    sfc_vals[sset] = {
        "P": P_sfc.mean(),
        "RH": RH_sfc.mean(),
        "T": T_sfc.mean(),
        "q": q_sfc.mean(),
    }

T_sfc = sfc_vals["rapsodi"]["T"] + 0.2
q_sfc = mt.partial_pressure_to_specific_humidity(
    sfc_vals["rapsodi"]["RH"] * es(T_sfc), sfc_vals["rapsodi"]["P"]
)
sfc_vals["rapsodi_p02"] = {
    "P": sfc_vals["rapsodi"]["P"],
    "RH": sfc_vals["rapsodi"]["RH"],
    "T": T_sfc,
    "q": q_sfc,
}

for sset in ["gate", "rapsodi", "rapsodi_p02"]:
    for key, val in sfc_vals[sset].items():
        print(f"{key}: {val.values:.5f}")

T_gate = sfc_vals["gate"]["T"].values
q_gate = sfc_vals["gate"]["q"].values
T_orce = sfc_vals["rapsodi"]["T"].values
q_orce = sfc_vals["rapsodi"]["q"].values
T_op02 = sfc_vals["rapsodi_p02"]["T"].values
q_op02 = sfc_vals["rapsodi_p02"]["q"].values
# %%
# - zero-degree isotherms
#
print(f"Height of 0º isotherm:")
for ds in [gs_PE, bs_PE, rs_PE]:
    Tx = (ds.ta - constants.T0) ** 2
    z_T0 = Tx.idxmin(dim="altitude")
    if ds.title[0] == "G":
        z_T0_gate = z_T0
    print(
        f"  {ds.title[: ds.title.find(' ')]:9s}: {z_T0.median().values:.2f}m ({z_T0.quantile(0.1).values:.2f} m,{z_T0.quantile(0.9).values:.2f} m)"
    )

# %%
# - cold point
#
print(f"Height and temerature of cold point:")
for ds in [gs_PE, rs_PE]:
    z_cp = ds.ta.idxmin(dim="altitude")
    t_cp = ds.ta.min(dim="altitude")
    print(
        f"  {ds.title[: ds.title.find(' ')]:9s}: {z_cp.median().values:.2f} m ({z_cp.quantile(0.1).values:.2f} m, {z_cp.quantile(0.9).values:.2f} m)"
    )
    print(
        f"            {t_cp.median().values:.2f} K ({t_cp.quantile(0.1).values:.2f} K, {t_cp.quantile(0.9).values:.2f} K)"
    )

sns.set_context(context="paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 2), sharey=True)
z_cp.plot.hist(bins=100, density=True, color="navy", xlim=(15000, 20000))
ax.set_xlabel("altitude / m")
sns.despine(offset=10)

Tx = rs_bar_PE.ta
RHice = svp.ice_wagner_etal(Tx) / svp.liq_wagner_pruss(Tx)
# %%
# - create theoretical soundings
#
P = np.arange(100900.0, 4000.0, -500)

r_consrv = mt_beta.mk_sounding_ds(P, T_orce, q_orce)
g_consrv = mt_beta.mk_sounding_ds(P, T_gate, q_gate)
p_pseudo = mt_beta.mk_sounding_ds(P, T_op02, q_op02, thx=mt.theta_e_bolton)
r_pseudo = mt_beta.mk_sounding_ds(P, T_orce, q_orce, thx=mt.theta_e_bolton)
g_pseudo = mt_beta.mk_sounding_ds(P, T_gate, q_gate, thx=mt.theta_e_bolton)
r_wthice = mt_beta.mk_sounding_ds(P, T_orce, q_orce, integrate=True)
g_wthice = mt_beta.mk_sounding_ds(P, T_gate, q_gate, integrate=True)

# %%
# - plot profiles
#
sns.set_context(context="talk")
fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

ylim = (0, 19000)
rlim = (0, 1)
dlim = (295, 355)

kwargs = {"ax": ax[0], "y": "altitude", "ylim": ylim, "xlim": (294, 404)}
rs_bar_PE.theta.plot(c=colors["rapsodi"], ls="-", label="rapsodi", **kwargs)
bs_bar_PE.theta.plot(c=colors["beach"], ls="-", label="beach", **kwargs)
gs_bar_PE.theta.plot(c=colors["gate"], ls="-", label="gate", **kwargs)

r_pseudo["theta"].plot(c="grey", ls="--", label="pseudo", **kwargs)
r_consrv["theta"].plot(c="lightgrey", ls="--", label="moist", **kwargs)
r_wthice["theta"].plot(c="grey", ls=":", label="ice", **kwargs)

kwargs = {"ax": ax[1], "y": "altitude", "ylim": ylim, "xlim": (0, 1)}
rs_bar_PE.rh.plot(c=colors["rapsodi"], ls="-", **kwargs)
bs_bar_PE.rh.plot(c=colors["beach"], ls="-", **kwargs)
gs_bar_PE.rh.plot(c=colors["gate"], ls="-", **kwargs)

ax[1].plot(
    [halo_rh.quantile(0.34), halo_rh.quantile(0.65)],
    [zbar, zbar],
    lw=2.5,
    c="k",
    label="HALO 35 to 65",
)
ax[1].plot(
    [halo_rh.quantile(0.1), halo_rh.quantile(0.9)],
    [zbar, zbar],
    lw=0.5,
    c="k",
    label="HALO 0 to 90",
)
ax[1].plot(
    [halo_rh.quantile(0.49), halo_rh.quantile(0.51)], [zbar, zbar], lw=2.5, c="w"
)
ax[1].legend(fontsize=9)

RHice.plot(c=colors["rapsodi"], ls="dotted", **kwargs, label="ice saturated")
ax[1].legend(fontsize=9)

kwargs = {"ax": ax[2], "y": "altitude", "ylim": ylim, "xlim": (0, 0.02)}
mt.get_n2(rs_bar_PE.theta, rs_bar_PE.q, rs_bar_PE.altitude, axis=0).plot(
    c=colors["rapsodi"], ls="-", label="rapsodi", **kwargs
)
mt.get_n2(bs_bar_PE.theta, bs_bar_PE.q, bs_bar_PE.altitude).plot(
    c=colors["beach"], ls="-", label="beach", **kwargs
)
mt.get_n2(gs_bar_PE.theta, gs_bar_PE.q, gs_bar_PE.altitude).plot(
    c=colors["gate"], ls="-", label="gate", **kwargs
)

mt.get_n2(r_pseudo["theta"], r_pseudo["q"], r_pseudo["altitude"]).plot(
    c="grey", ls="--", label="pseudo", **kwargs
)
mt.get_n2(r_consrv["theta"], r_consrv["q"], r_consrv["altitude"]).plot(
    c="lightgrey", ls="--", label="moist", **kwargs
)
mt.get_n2(r_wthice["theta"], r_wthice["q"], r_consrv["altitude"]).plot(
    c="grey", ls=":", label="ice", **kwargs
)

ax[2].set_xlabel("$N$ / s")
ax[2].set_xticks([0, 0.01, 0.02])
ax[2].set_ylabel("")

ax[0].set_xlabel("$\\theta$ / K")
ax[0].set_ylabel("altitude / m")
ax[0].set_xticks([300, 330, 360, 390])
ax[0].set_yticks(np.arange(0, 18001, 3000))

ax[1].set_xlabel("RH")
ax[1].set_ylabel("")

ax[1].set_yticks([z_cp.quantile(0.35), z_cp.quantile(0.35)], minor="True")
ax[1].tick_params(axis="both", which="minor", colors="red")

ax[0].legend(fontsize=10)
fig.tight_layout()
sns.despine(offset=10)
plt.savefig("plots/sounding.pdf")

# %%
# -- plot differencex wrt GATE
#
ylim = (0, 17000)
tlim = (294, 360)
nlim = (0, 0.02)
rlim = (0, 1)
dlim = (295, 355)
x = r_pseudo["T"]
x = x - g_pseudo["T"].values
x = x.where(x > 0.01)

y = r_pseudo["T"]
y = y - g_pseudo["T"].values
y = y.where(y > 0.01)

dtheta_rs = rs_bar_PE.theta - gs_bar_PE.theta
dtheta_bs = bs_bar_PE.theta - gs_bar_PE.theta

sns.set_context("paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 4), sharey=True)

dtheta_rs.plot(
    ax=ax, y="altitude", ylim=ylim, xlim=dlim, label="rapsodi", color=colors["rapsodi"]
)
dtheta_bs.plot(
    ax=ax, y="altitude", ylim=ylim, xlim=dlim, label="beach", color=colors["beach"]
)

x.plot(ax=ax, y="altitude", color="k", ls="dotted", label="theory", lw=1)
y.plot(ax=ax, y="altitude", color="k", ls="dotted", lw=1)

ax.plot(
    np.asarray([0, 3.5]),
    np.asarray([z_T0.quantile(0.5), z_T0.quantile(0.5)]),
    color="grey",
    lw="0.5",
    ls="-",
)
ax.annotate(
    "$z_0$",
    xy=(-1, z_T0.quantile(0.5)),
    fontsize=8,
)

ax.set_xlim(-5, 6)
ax.set_ylim(0, 22500)
ax.set_xlabel("$\\Delta \\theta$ / K")
ax.set_ylabel("altitude / m")
ax.set_ylabel("altitude / km")

dth_st = dtheta_rs.sel(altitude=slice(20000, 22000)).mean().values
dth_bl = dtheta_rs.sel(altitude=slice(200, 600)).mean().values
print(dth_bl)

ax.set_xticks(
    [
        np.round(dtheta_rs.min().values, 1),
        np.round(dth_st, 1),
        np.round(dth_bl, 1),
        np.round(dtheta_rs.max().values, 1),
    ]
)
ax.set_xticks([0], minor=True)

ax.set_yticks(np.arange(0, 24000, 3000))
ax.set_yticklabels([0, 3, 6, 9, 12, 15, 18, 21])
ax.set_yticks([z_T0_gate.quantile(0.5), z_T0.quantile(0.5)], minor=True)

plt.legend(loc="best", fontsize=8)
plt.tight_layout()
sns.despine(offset=10)
plt.savefig("plots/DeltaT.pdf")
# %%
# -- plot difference depending on domain
#
sns.set_context("paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 4), sharey=True)

label = "beach"
delta_bs = bs_GA_bar.theta - bs_bar_PE.theta
delta_bar_bs = delta_bs.sel(altitude=slice(0, 14000)).mean().values
delta_bs.plot(ax=ax, y="altitude", ylim=ylim, label=label)
delta_bs

label = "rapsodi"
delta_rs = rs_GA_bar.theta - rs_bar_PE.theta
delta_bar_rs = delta_rs.sel(altitude=slice(0, 14000)).mean().values
delta_rs.plot(ax=ax, y="altitude", ylim=ylim, label=label)

ax.set_ylim(0, 14300)
ax.set_xlim(-1, 1)
ax.set_xlabel("$\\Delta \\theta$ / K")
ax.set_ylabel("altitude / m")
ax.set_ylabel("altitude / km")

ax.set_xticks([delta_bar_bs, delta_bar_rs], minor=True)

ax.set_yticks(np.arange(0, 12300, 3000))
ax.set_yticklabels([0, 3, 6, 9, 12])

plt.legend()
sns.despine(offset=10)
plt.savefig("plots/DeltaT2.pdf")
print(f"differences: beach {delta_bar_bs:.2f}, rapsodi {delta_bar_rs:.2f}")
# %%
# - look at distributions
#
sns.set_context("paper")
fig, ax = plt.subplots(1, 3, figsize=(10, 3))

gate["theta"].sel(altitude=slice(0, 300)).plot.hist(
    ax=ax[0], bins=300, alpha=0.4, density=True, color=colors["gate"]
)

rapsodi["theta"].sel(altitude=slice(0, 300)).plot.hist(
    ax=ax[0], bins=30, alpha=0.4, density=True, color=colors["rapsodi"]
)

beach["theta"].sel(altitude=slice(0, 300)).plot.hist(
    ax=ax[0], bins=30, alpha=0.4, density=True, color=colors["beach"]
)

ax[0].set_xlim(294, 304)
ax[0].set_xlabel("$\\theta_\\mathrm{bl}$ / K")

x1 = (
    rapsodi["theta"]
    .sel(altitude=slice(0, 600))
    .stack(points=["sonde", "altitude"])
    .quantile(0.5)
    .values
)
x2 = (
    gate["theta"]
    .sel(altitude=slice(0, 600))
    .stack(points=["sonde", "altitude"])
    .quantile(0.5)
    .values
)

ax[0].set_xticks([np.round(x1, 2), np.round(x2, 2)])

gate["rh"].sel(altitude=slice(10, 100)).plot.hist(
    ax=ax[1], bins=300, alpha=0.4, density=True, color=colors["gate"]
)
rapsodi["rh"].sel(altitude=slice(10, 100)).plot.hist(
    ax=ax[1], bins=30, alpha=0.4, density=True, color=colors["rapsodi"]
)
beach["rh"].sel(altitude=slice(10, 100)).plot.hist(
    ax=ax[1], bins=30, alpha=0.4, density=True, color=colors["beach"]
)
ax[1].set_xlim(0.5, 1.1)
ax[1].set_xlabel("RH$_\\mathrm{sfc}$ / $-$")

x1 = (
    rapsodi["rh"]
    .sel(altitude=slice(0, 50))
    .stack(points=["sonde", "altitude"])
    .quantile(0.5)
    .values
)
x2 = (
    gate["rh"]
    .sel(altitude=slice(0, 50))
    .stack(points=["sonde", "altitude"])
    .quantile(0.5)
    .values
)

ax[1].set_xticks([0.75, np.round((x1 + x2) / 2, 2), 0.9])

gate["p"].sel(altitude=slice(10, 50)).plot.hist(
    ax=ax[2], bins=30, alpha=0.4, density=True, color=colors["gate"]
)
rapsodi["p"].sel(altitude=slice(10, 50)).plot.hist(
    ax=ax[2], bins=30, alpha=0.4, density=True, color=colors["rapsodi"]
)
beach["p"].sel(altitude=slice(10, 50)).plot.hist(
    ax=ax[2], bins=30, alpha=0.4, density=True, color=colors["beach"]
)
ax[2].set_xlim(100000, 102000)
ax[2].set_xlabel("$p_\\mathrm{sfc}$ / $-$")

x1 = (
    rapsodi["p"]
    .sel(altitude=slice(0, 50))
    .stack(points=["sonde", "altitude"])
    .quantile(0.5)
    .values
)
x2 = (
    gate["p"]
    .sel(altitude=slice(0, 50))
    .stack(points=["sonde", "altitude"])
    .quantile(0.5)
    .values
)

ax[2].set_xticks([100000, np.round((x1 + x2) / 2, 2), 102000])

sns.despine(offset=10)
# %%
# - plot theta profiles
#
sns.set_context(context="paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharey=True)

ylim = (0, 15000)
tlim = (296, 356)
nlim = (0, 0.02)
rlim = (0, 1)

rs_bar_PE.theta.plot(
    ax=ax, y="altitude", ylim=ylim, xlim=tlim, label="rapsodi", color=colors["rapsodi"]
)
bs_bar_PE.theta.plot(
    ax=ax, y="altitude", ylim=ylim, xlim=tlim, label="beach", color=colors["beach"]
)
gs_bar_PE.theta.plot(
    ax=ax, y="altitude", ylim=ylim, xlim=tlim, label="beach", color=colors["gate"]
)

ax.set_xlabel("$\\theta$ / K")
ax.set_ylabel("altitude / m")

ax.legend(fontsize=10)
fig.tight_layout()
sns.despine(offset=10)
plt.savefig("plots/theta.pdf")
# %%
# - plot zonal winds
#
sns.set_context(context="paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharey=True)

rs_bar_PE.u.plot(
    ax=ax, y="altitude", ylim=ylim, label="rapsodi", color=colors["rapsodi"]
)
gs_bar_PE.u.plot(ax=ax, y="altitude", ylim=ylim, label="beach", color=colors["gate"])
bs_bar_PE.u.plot(
    ax=ax, y="altitude", ylim=(0, 25000), label="gate", color=colors["beach"]
)
ax.set_xlabel("$u$ / ms$^{-1}$")
ax.set_ylabel("altitude / m")
ax.set_xlim(-23, 7)

sns.despine(offset=10)
plt.savefig("plots/zonal-wind.pdf")
