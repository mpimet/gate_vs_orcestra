#  %%
# % pip install moist_thermodynamics
# -------------
# Define some functions for plotting and evaluating atmospheric soundings
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.path import Path
from moist_thermodynamics import functions as mt
from moist_thermodynamics import saturation_vapor_pressures as svp
from moist_thermodynamics import constants

es = svp.liq_wagner_pruss
P0 = constants.P0
Rd = constants.Rd
Rv = constants.Rv

kappa = constants.Rd / constants.cpd

gate_A = np.array(
    [
        [-27.0, 6.5],
        [-23.5, 5.0],
        [-20.0, 6.5],
        [-20.0, 10.5],
        [-23.5, 12.0],
        [-27.0, 10.5],
    ]
)

percusion_E = np.array([[-34.0, 13.5], [-20.0, 13.5], [-20.0, 3.5], [-34.0, 3.5]])


def get_n2(th, qv):
    """Returns the Brunt-Vaisala frequeny for unsaturated air.

    It assumes that the input are type xarray with their first coordinate being
    altitude in meters, and that the air is no where saturated

    Args:
        th: potential temperature
        qv: specific humidity
    """

    from moist_thermodynamics import constants

    Rv = constants.water_vapor_gas_constant
    Rd = constants.dry_air_gas_constant
    g = constants.gravity_earth
    R = Rd + (Rv - Rd) * qv
    dlnthdz = np.log(th).differentiate(th.dims[0])
    dqvdz = qv.differentiate(th.dims[0])

    return np.sqrt(g * (dlnthdz + (Rv - Rd) / R * dqvdz))


def get_adiabat(P, Tsfc=301.0, qsfc=17e-3, Tmin=190.0, thx=mt.theta_l, integrate=False):
    """Returns the moist adiabat along a pressure dimension.

    Cacluates the moist adiabate based either on an integration or a specified
    isentrope with pressure as the vertical coordinate.

    Args:
        P: pressure
        Tsfc: temperature at greatest pressure level
        RH: relative humidity at greatest pressure level
        Tmin: minimum temperature of adiabat
        thx: function to calculate isentrope
        integrate: determines if explicit integration will be used.
    """

    from moist_thermodynamics import constants
    from moist_thermodynamics import saturation_vapor_pressures as svp

    es = svp.liq_analytic
    T0 = constants.T0
    i4T = np.vectorize(mt.invert_for_temperature)

    Tx = thx(Tsfc, P[0], qsfc)
    TK = i4T(thx, Tx, P, qsfc)

    if integrate:
        es = mt.make_es_mxd(es_liq=svp.liq_analytic, es_ice=svp.ice_analytic)
        dP = P[1] - P[0]
        Ptop = P[-1] + dP
        Tice, Py = mt.moist_adiabat(
            Tsfc,
            P[0],
            Ptop,
            dP,
            qsfc,
            cc=constants.ci,
            lv=mt.sublimation_enthalpy,
            es=es,
        )
        Tliq, Px = mt.moist_adiabat(
            Tsfc,
            P[0],
            Ptop,
            dP,
            qsfc,
            cc=constants.cl,
            lv=mt.vaporization_enthalpy,
            es=es,
        )
        TK = np.ones(len(Px)) * T0
        TK[Tliq > T0] = Tliq[Tliq > T0]
        TK[Tice < T0] = Tice[Tice < T0]

    T = xr.DataArray(
        np.maximum(TK, Tmin),
        dims=("P",),
        coords=[P],
        attrs={"units": "K", "long_name": "air temperature", "symbol": "$T$"},
    )

    return T


def hydrostatic_altitude(s):
    """Returns height from hydrostatic integration of state

    Hydrostatically integrates the thermodynamic state to derive an altitude
    coordinate and then returns the state with this new variable as the
    dimension

    Args:
       state: xarray dataset with P, T and q
    """
    from moist_thermodynamics import constants

    Rv = constants.water_vapor_gas_constant
    Rd = constants.dry_air_gas_constant
    g = constants.gravity_earth

    qbar = ((s.q + s.q.shift(P=+1)) / 2).isel(P=slice(1, None))
    Tbar = ((s.T + s.T.shift(P=+1)) / 2).isel(P=slice(1, None))
    dz = -(Rd + (Rv - Rd) * qbar) * Tbar * np.log(s.P).diff(dim="P") / g
    altitude = xr.DataArray(
        np.zeros(s.T.size),
        dims=s.T.dims,
        coords=s.T.coords,
        attrs={"units": "m", "long_name": "altitude", "symbol": "$z$"},
    )
    altitude.loc[{"P": s.P.isel(P=slice(1, None))}] = dz.cumsum()

    return altitude


def p2z_coord(ds: xr.Dataset) -> xr.Dataset:
    """
    change from pressure to height coordinaes
    """
    return (
        ds.assign_coords({"altitude": hydrostatic_altitude(ds)})
        .swap_dims({"P": "altitude"})
        .copy(deep=True)
    )


def sub_domain(ds, polygon):
    """
    select points from dataset that lie within the polygon
    """
    points = np.column_stack([ds.longitude.values, ds.latitude.values])
    inside = Path(polygon).contains_points(points)
    return ds.sel(sonde=inside)


def reformat_rs(ds):
    """
    standarize naming for rapsodi data
    """
    ds = (
        ds.reset_coords("p", drop=False)
        .swap_dims({"launch_time": "sounding"})
        .rename_dims({"alt": "altitude","sounding": "sonde"})
        .rename_vars(
            {
                "launch_lat": "latitude",
                "launch_lon": "longitude",
                "launch_time": "time",
                "alt": "altitude",
                "sounding": "sonde"
            }
        )
        .reset_coords(["flight_lat", "flight_lon", "flight_time"], drop=False)
        .drop_vars("sonde")
    )

    return ds


def reformat_ds(ds):
    """
    standarize naming for beach data
    """
    ds = (
        ds.rename_vars(
            {"aircraft_latitude": "latitude", "aircraft_longitude": "longitude"}
        )
        .reset_coords("aircraft_msl_altitude", drop=False)
        .rename_vars({"sonde_time": "time"})
    )

    return ds


def reformat_gs(ds):
    """
    standarize naming for gate data
    """
    ds = (
        ds.set_coords(["latitude", "longitude"])
        .assign_coords({"sonde": ds.src})
        .sel(time=slice("1974-08-10", "1974-09-30"))
        .swap_dims({"time": "sonde"})
        .drop_vars("sonde")
    )
    return ds


# %%
# - load data
#
beach = xr.open_zarr(
    "ipfs://bafybeicb33v6ohezyhgq5rumq4g7ejnfqxzcpuzd4i2fxnlos5d7ebmi3m"
).pipe(reformat_ds)
rapsodi = xr.open_zarr(
    "ipfs://bafybeigensqyqxfyaxgyjhwn6ytdpi3i4sxbtffd4oc27zbimyro4hygjq"
).pipe(reformat_rs)
gate = xr.open_zarr("ipfs://QmY44nwC5dNUnZYFs5h1XA4eJ24RioeS9mHzutnhgYmtzM").pipe(
    reformat_gs
)
rapsodi
# %%
# - localize data into different domains
#
gs1 = sub_domain(gate, gate_A)
ds1 = sub_domain(beach, gate_A)
rs1 = sub_domain(rapsodi, gate_A)

gs2 = sub_domain(gate, percusion_E)
ds2 = sub_domain(beach, percusion_E)
rs2 = sub_domain(rapsodi, percusion_E)
# %%
# - parameters for soundings
#
sfc_vals = {}
sondes = {"gate": gs1, "rapsodi": rs1}
for sset in ["gate", "rapsodi"]:
    print (sset)
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
z0_ds = (
    ds1["altitude"]
    .where((ds1.ta.mean(dim="sonde") < 273.15).compute(), drop=True)[0]
    .values
)
z0_rs = (
    rs1["altitude"]
    .where((rs1.ta.mean(dim="sonde") < 273.15).compute(), drop=True)[0]
    .values
)
z0_gate = (
    gs1["altitude"]
    .where((gs1.ta.mean(dim="sonde") < 273.15).compute(), drop=True)
    .values
)

print(
    f"Freezing levels:\n ORCESTRA radiosondes {z0_rs:.1f}m,\n ORCESTRA dropsondes {z0_ds:.1f}m,\n GATE {z0_gate[0]:.1f}m"
)
# %%
# - create idealized soundings
#
P = np.arange(100900, 4000, -500)

TPq = xr.Dataset(
    data_vars={
        "T": (("P",), np.full(len(P), T_gate)),
        "q": (("P",), np.full(len(P), q_gate)),
    },
    coords={
        "P": ("P", P, {"units": "Pa", "long_name": "air pressure", "symbol": "$P$"})
    },
)
TPq["T"].attrs = {"units": "K", "long_name": "Temperature", "symbol": "$T$"}
TPq["q"].attrs = {"units": "Pa", "long_name": "air pressure", "symbol": "$P$"}

TPq["T"] = get_adiabat(P, Tsfc=T_orce, qsfc=q_orce)
consrv = TPq.pipe(p2z_coord)

TPq["T"] = get_adiabat(P, Tsfc=T_gate, qsfc=q_gate)
consrv_gate = TPq.pipe(p2z_coord)

TPq["T"] = get_adiabat(P, Tsfc=T_op02, qsfc=q_op02, thx=mt.theta_e_bolton)
pseudo02 = TPq.pipe(p2z_coord)

TPq["T"] = get_adiabat(P, Tsfc=T_orce, qsfc=q_orce, thx=mt.theta_e_bolton)
pseudo = TPq.pipe(p2z_coord)

TPq["T"] = get_adiabat(P, Tsfc=T_gate, qsfc=q_gate, thx=mt.theta_e_bolton)
pseudo_gate = TPq.pipe(p2z_coord)

TPq["T"] = get_adiabat(P, Tsfc=T_orce.item(), qsfc=q_orce.item(), integrate=True)
wthice = TPq.pipe(p2z_coord)

TPq["T"] = get_adiabat(P, Tsfc=T_gate.item(), qsfc=q_gate.item(), integrate=True)
wthice_gate = TPq.pipe(p2z_coord)

for s in [consrv, pseudo, wthice]:
    s["q"] = ds1.mean(dim="sonde").q.interp(altitude=s.altitude)
    s["theta"] = s["T"] * (P0 / s["P"]) ** kappa
# %%
# - profiles
#
gs = gs1.mean(dim="sonde").coarsen(altitude=5, boundary="trim").mean()
ds = ds1.mean(dim="sonde").coarsen(altitude=5, boundary="trim").mean()
rs = rs1.mean(dim="sonde").coarsen(altitude=5, boundary="trim").mean()

sns.set_context(context="talk")
fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

ylim = (0, 17000)
tlim = (294, 360)
nlim = (0, 0.02)
rlim = (0, 1)
dlim = (295, 355)

rs.theta.plot(ax=ax[0], y="altitude", ylim=ylim, xlim=tlim, label="rapsodi")
ds.theta.plot(ax=ax[0], y="altitude", ylim=ylim, xlim=tlim, label="beach")
gs.theta.plot(ax=ax[0], y="altitude", ylim=ylim, xlim=tlim, label="gate")

rs.rh.plot(ax=ax[1], y="altitude", ylim=ylim, xlim=rlim, label="rapsodi")
ds.rh.plot(ax=ax[1], y="altitude", ylim=ylim, xlim=rlim, label="beach")
gs.rh.plot(ax=ax[1], y="altitude", ylim=ylim, xlim=rlim, label="gate")

get_n2(rs.theta, rs.q).plot(
    ax=ax[2], y="altitude", ylim=ylim, xlim=nlim, label="rapsodi"
)
get_n2(ds.theta, ds.q).plot(ax=ax[2], y="altitude", ylim=ylim, xlim=nlim, label="beach")
get_n2(gs.theta, gs.q).plot(ax=ax[2], y="altitude", ylim=ylim, xlim=nlim, label="beach")

pseudo["theta"].plot(
    ax=ax[0],
    y="altitude",
    ylim=ylim,
    xlim=tlim,
    color="k",
    ls="dashed",
    lw="1",
    label="pseudo",
)
consrv["theta"].plot(
    ax=ax[0],
    y="altitude",
    ylim=ylim,
    xlim=tlim,
    color="grey",
    ls="dashed",
    lw="1",
    label="moist",
)
wthice["theta"].plot(
    ax=ax[0],
    y="altitude",
    ylim=ylim,
    xlim=tlim,
    color="grey",
    ls="dotted",
    lw="1",
    label="moist w/ice",
)

get_n2(wthice["theta"], wthice["q"]).plot(
    ax=ax[2],
    y="altitude",
    ylim=ylim,
    xlim=nlim,
    color="grey",
    ls="dotted",
    lw="1",
    label="with ice",
)
get_n2(consrv["theta"], consrv["q"]).plot(
    ax=ax[2],
    y="altitude",
    ylim=ylim,
    xlim=nlim,
    color="grey",
    ls="dashed",
    lw="1",
    label="moist adiabat",
)
get_n2(pseudo["theta"], pseudo["q"]).plot(
    ax=ax[2],
    y="altitude",
    ylim=ylim,
    xlim=nlim,
    color="k",
    ls="dashed",
    lw="1",
    label="pseudo adiabat",
)

ax[2].set_xlabel("$N$ / s")
ax[2].set_xticks([0, 0.01, 0.02])
ax[2].set_ylabel("")

ax[0].set_xlabel("$\\theta$ / K")
ax[0].set_ylabel("altitude / m")
ax[0].set_xticks([300, 330, 360])

ax[1].set_xlabel("RH")
ax[1].set_ylabel("")

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
x = pseudo["T"]
x = x - pseudo_gate["T"].values
x = x.where(x > 0.01)

y = pseudo02["T"]
y = y - pseudo_gate["T"].values
y = y.where(y > 0.01)

dtheta_rs = rs.theta - gs.theta
dtheta_ds = ds.theta - gs.theta

sns.set_context("paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 4), sharey=True)

dtheta_rs.plot(ax=ax, y="altitude", ylim=ylim, xlim=dlim, label="rapsodi", color="navy")
dtheta_ds.plot(ax=ax, y="altitude", ylim=ylim, xlim=dlim, label="beach", color="teal")

x.plot(ax=ax, y="altitude", color="k", ls="dotted", label="theory", lw=1)
y.plot(ax=ax, y="altitude", color="k", ls="dotted", lw=1)

ax.plot(
    np.asarray([0, 3.5]), np.asarray([z0_rs, z0_rs]), color="grey", lw="0.5", ls="-"
)
ax.annotate(
    "$z_0$",
    xy=(-1, z0_rs),
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
ax.set_yticks([z0_gate[0], z0_rs], minor=True)

plt.legend(loc="best", fontsize=8)
plt.tight_layout()
sns.despine(offset=10)
plt.savefig("plots/DeltaT.pdf")
# %%
# -- plot difference depending on domain
#
sns.set_context("paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 4), sharey=True)

delta_ds = (
    ds2.mean(dim="sonde").coarsen(altitude=5, boundary="trim").mean().theta - ds.theta
)
label = "beach"
delta_bar_ds = delta_ds.sel(altitude=slice(0, 14000)).mean().values
delta_ds.plot(ax=ax, y="altitude", ylim=ylim, label=label)
delta_ds

delta_rs = (
    rs2.mean(dim="sonde").coarsen(altitude=5, boundary="trim").mean().theta - rs.theta
)
label = "rapsodi"
delta_bar_rs = delta_rs.sel(altitude=slice(0, 14000)).mean().values
delta_rs.plot(ax=ax, y="altitude", ylim=ylim, label=label)

ax.set_ylim(0, 14300)
ax.set_xlim(-1, 1)
ax.set_xlabel("$\\Delta \\theta$ / K")
ax.set_ylabel("altitude / m")
ax.set_ylabel("altitude / km")

ax.set_xticks([delta_bar_ds, delta_bar_rs], minor=True)

ax.set_yticks(np.arange(0, 12300, 3000))
ax.set_yticklabels([0, 3, 6, 9, 12])

plt.legend()
sns.despine(offset=10)
plt.savefig("plots/DeltaT2.pdf")
print(f"differences: beach {delta_bar_ds:.2f}, rapsodi {delta_bar_rs:.2f}")
# %%
# - look at distributions
#
sns.set_context("paper")
fig, ax = plt.subplots(1, 3, figsize=(10, 3))

gate["theta"].sel(altitude=slice(0, 300)).plot.hist(
    ax=ax[0], bins=300, alpha=0.4, density=True, color="orangered"
)

rapsodi["theta"].sel(altitude=slice(0, 300)).plot.hist(
    ax=ax[0], bins=30, alpha=0.4, density=True, color="navy"
)

beach["theta"].sel(altitude=slice(0, 300)).plot.hist(
    ax=ax[0], bins=30, alpha=0.4, density=True, color="teal"
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
    ax=ax[1], bins=300, alpha=0.4, density=True, color="orangered"
)
rapsodi["rh"].sel(altitude=slice(10, 100)).plot.hist(
    ax=ax[1], bins=30, alpha=0.4, density=True, color="navy"
)
beach["rh"].sel(altitude=slice(10, 100)).plot.hist(
    ax=ax[1], bins=30, alpha=0.4, density=True, color="teal"
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
    ax=ax[2], bins=30, alpha=0.4, density=True, color="orangered"
)
rapsodi["p"].sel(altitude=slice(10, 50)).plot.hist(
    ax=ax[2], bins=30, alpha=0.4, density=True, color="navy"
)
beach["p"].sel(altitude=slice(10, 50)).plot.hist(
    ax=ax[2], bins=30, alpha=0.4, density=True, color="teal"
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
# - plot relative humidity and rh vs ice
#
es_liq = svp.liq_analytic
es_ice = svp.liq_analytic
es = mt.make_es_mxd(es_liq=svp.liq_analytic, es_ice=svp.ice_analytic)

rs_p = (rs.ta / rs.theta) ** (1 / kappa) * P0
R = Rd + (Rv - Rd) * rs.q
pv = rs.q * Rv / R * rs_p
rh = pv / es(rs.ta)

sns.set_context(context="paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharey=True)

rh.plot(
    y="altitude",
    xlim=(0, 1),
    ylim=(0, 23000),
    label="ice-liq",
    color="navy",
    ls="dotted",
)
rs.rh.plot(y="altitude", xlim=(0, 1), ylim=(0, 23000), label="liq", color="navy")
ax = plt.gca()
ax.set_ylabel("altitude / m")
ax.set_xlabel("RH")
sns.despine(offset=10)
plt.legend()

plt.savefig("plots/rh.pdf")
# %%
# - plot theta profiles
#
sns.set_context(context="paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharey=True)

ylim = (0, 12000)
tlim = (-1, 1)
nlim = (0, 0.02)
rlim = (0, 1)

y1 = rapsodi.mean(dim="sonde").theta
y2 = beach.mean(dim="sonde").theta
dy = y2[: len(y1)] - y1
dy.plot(ax=ax, y="altitude", label="$\\Delta \\Theta$")

y1 = rapsodi.mean(dim="sonde").ta
y2 = beach.mean(dim="sonde").ta
dy = y2[: len(y1)] - y1
dy.plot(ax=ax, y="altitude", label="$\\Delta T$")

ax.set_ylim(0, 12000)
ax.set_xlim(-0.6, 0.6)
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

rs.u.plot(ax=ax, y="altitude", ylim=ylim, label="rapsodi", color="navy")
gs.ua.plot(ax=ax, y="altitude", ylim=ylim, label="beach", color="orangered")
ds.u.plot(ax=ax, y="altitude", ylim=(0, 25000), label="gate", color="teal")
ax.set_xlabel("$u$ / ms$^{-1}$")
ax.set_ylabel("altitude / m")
ax.set_xlim(-30, 10)

sns.despine(offset=10)
plt.savefig("plots/zonal-wind.pdf")
# %%
# - compare deviations in altitude coordinate from hydrostaticity
#
sns.set_context(context="paper")
fig, ax = plt.subplots(1, 1, figsize=(5, 3), sharey=True)

z_r = (
    constants.gravity_earth
    / rapsodi.mean(dim="sonde").ta
    / (Rd + (Rv - Rd) * rapsodi.mean(dim="sonde").q)
)
z_b = (
    constants.gravity_earth
    / beach.mean(dim="sonde").ta
    / (Rd + (Rv - Rd) * beach.mean(dim="sonde").q)
)

ddz_r = (
    rapsodi.altitude.diff(dim="altitude")
    + np.log(rapsodi.mean(dim="sonde").p).diff(dim="altitude")
    / z_r.rolling(altitude=2).mean()
)
ddz_b = (
    beach.altitude.diff(dim="altitude")
    + np.log(beach.mean(dim="sonde").p).diff(dim="altitude")
    / z_b.rolling(altitude=2).mean()
)

ddz_r.coarsen(altitude=10, boundary="trim").mean().plot(
    color="navy", label=f"rapsodi {ddz_r.mean().values:.2f} m"
)
ddz_b.coarsen(altitude=10, boundary="trim").mean().plot(
    color="teal", label=f"beach   {ddz_b.mean().values:.2f} m"
)

print(y1.mean().values)
ax.set_ylabel("$\\mathrm{d}z + R\\overline{T}g^{-1} \\, \\mathrm{d} \\,\\ln p$ / m")
ax.set_xlabel("altitude / m")
ax.set_ylim(-0.15, 0.15)
ax.set_xlim(0, 12000)
plt.legend()

sns.despine(offset=10)