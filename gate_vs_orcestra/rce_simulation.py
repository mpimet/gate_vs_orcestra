import tempfile

import konrad
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def get_atmosphere(co2, nlev=256):
    plev, phlev = konrad.utils.get_pressure_grids(1000e2, 1, nlev)
    atmosphere = konrad.atmosphere.Atmosphere(phlev)
    atmosphere["CO2"][:] = co2

    return atmosphere


def run_rce(co2=400e-6, sst=300):
    """Run RCE at given CO2 concentration and fixed SST."""
    outfile = tempfile.NamedTemporaryFile(delete=False, suffix=".nc").name

    rce = konrad.RCE(
        atmosphere=get_atmosphere(co2=co2),
        surface=konrad.surface.FixedTemperature(temperature=sst),
        # lapserate=konrad.lapserate.BoundaryLayer(),
        humidity=konrad.humidity.FixedRH(konrad.humidity.Manabe67()),
        timestep="3h",
        max_duration="150d",
        outfile=outfile,
    )
    rce.run()

    ds = xr.merge(
        [
            xr.open_dataset(outfile, group="atmosphere"),
            xr.open_dataset(outfile, group="convection"),
            xr.open_dataset(outfile, group="surface"),
            xr.open_dataset(outfile, group="radiation"),
        ],
    )

    ds = ds.assign(
        cold_point_height=rce.atmosphere["z"][-1, rce.atmosphere.get_cold_point_index()]
    )

    return ds


# GATE
gate_co2 = 303.5e-6
gate = run_rce(co2=gate_co2, sst=300.0)

# ORCESTRA
orcestra_co2 = 422.8e-6
orcestra_co2_e = np.exp(1.5 * np.log(orcestra_co2 / gate_co2)) * gate_co2
orcestra = run_rce(co2=orcestra_co2_e, sst=301.5)


gate["lw_flxu_clr"][-1, -1].values, orcestra["lw_flxu_clr"][-1, -1].values


alt = np.linspace(0, 40_000, 2**7)

gate_z = (
    gate.assign_coords(z=gate.z[-1]).swap_dims(plev="z").interp(z=alt, method="pchip")
)
orcestra_z = (
    orcestra.assign_coords(z=orcestra.z[-1])
    .swap_dims(plev="z")
    .interp(z=alt, method="pchip")
)
diff_z = orcestra_z - gate_z

fig, ax = plt.subplots(figsize=(6, 6))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.axvline(0, c="k", lw=0.8)

ax.plot(diff_z["T"].isel(time=-1), alt / 1000.0, color="#333")

horizontal_marks = (
    (gate["cold_point_height"] / 1000.0, "tab:blue"),
    (gate["convective_top_height"][-1] / 1000.0, "tab:red"),
    (orcestra["cold_point_height"] / 1000.0, "tab:blue"),
    (orcestra["convective_top_height"][-1] / 1000.0, "tab:red"),
)

for z, c in horizontal_marks:
    ax.axhline(
        z,
        color=c,
        lw=0.8,
        zorder=0,
    )


ax.set_xlim(-10, 7.5)
ax.set_xticks(
    [
        -10,
        np.round(diff_z["T"].min().values, 1),
        0,
        np.round((orcestra - gate).temperature.isel(time=-1), 1),
        np.round(diff_z["T"].max().values, 1),
    ]
)
ax.set_xlabel("$\\Delta T$ / K")
ax.set_ylim(0, alt.max() / 1000)
ax.set_yticks(
    [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
    ]
)
ax.set_ylabel("$z$ / km")
fig.savefig("gate_vs_orcestra_rce.png")


diff_z["T"][-1].sel(z=slice(20e3, 30e3)).mean().values


[h[0].values * 1000 for h in horizontal_marks]
