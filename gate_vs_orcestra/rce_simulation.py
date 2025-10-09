# %%
import tempfile

import konrad
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def get_ozone_profile(p, period):
    """Return an ozone profile in pressure coordinates.

    Parameters:
        p (ndarray): Pressure coordinate to interpolate to
        period (str): Time period ("gate" or "orcestra")
    """
    match period.lower():
        case "gate":
            ds = xr.open_dataset("../data/mean_GATE_v0_east_GATE_sd_bg_int.nc")
        case "orcestra":
            ds = xr.open_dataset("../data/mean_RAPSODI_L2_v0_east_sd_bg_int.nc")

    ds = (
        ds.squeeze()
        .assign_coords(p_lay=(("z_lay",), ds.p_lay.squeeze().values))
        .swap_dims(z_lay="p_lay")
    )

    return ds.interp(p_lay=p)["o3_mmr"] * 28.9647 / 47.9982  # mmr -> vmr


def get_atmosphere(co2, o3, nlev=256):
    """Create a konrad atmosphere.

    Parameters:
        co2 (float): CO2 volume mixing ratio
        o3 (str): Ozone profile ("gate" or "orcestra")
        nlev (int): Number of pressure levels
    """
    plev, phlev = konrad.utils.get_pressure_grids(1000e2, 1, nlev)
    atmosphere = konrad.atmosphere.Atmosphere(phlev)

    atmosphere["CO2"][:] = co2
    atmosphere["O3"][0] = get_ozone_profile(plev, o3.lower())

    return atmosphere


def run_rce(co2, o3, sst):
    """Run RCE at given CO2 concentration and fixed SST."""
    # Store output to tempfile.
    outfile = tempfile.NamedTemporaryFile(delete=False, suffix=".nc").name

    rce = konrad.RCE(
        atmosphere=get_atmosphere(co2=co2, o3=o3),
        surface=konrad.surface.FixedTemperature(temperature=sst),
        humidity=konrad.humidity.FixedRH(konrad.humidity.Manabe67()),
        timestep="1h",
        max_duration="150d",
        outfile=outfile,
    )
    rce.run()

    # Open relevant netCDF groups
    ds = xr.merge(
        [
            xr.open_dataset(outfile, group="atmosphere"),
            xr.open_dataset(outfile, group="convection"),
            xr.open_dataset(outfile, group="surface"),
            xr.open_dataset(outfile, group="radiation"),
        ],
    ).assign(
        cold_point_height=rce.atmosphere["z"][-1, rce.atmosphere.get_cold_point_index()]
    )

    return ds


# +
def interpolate_altitude(ds, start=0, stop=40_000, num=2**7):
    """Interpolate konrad results from pressure to altitude coordinates

    Parameters:
        ds (xr.Dataset): konrad results
        start (float): Lowest altitude
        stop (float): Highest altitude
        num (int): Number of altitude levels
    """
    alt = np.linspace(start, stop, num)

    return (
        ds.assign_coords(z=ds.z[-1]).swap_dims(plev="z").interp(z=alt, method="pchip")
    )


def plot_comparison(gate, orcestra):
    """Compate two konrad results."""
    gate_z = interpolate_altitude(gate)
    orcestra_z = interpolate_altitude(orcestra)

    alt = gate_z.z.values
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

    return fig, ax


def print_changes(gate, orcestra):
    """Print change in cold-point/convective top as Markdown table."""
    gate_cp = gate["cold_point_height"].values
    gate_ct = gate["convective_top_height"][-1].values
    orcestra_cp = orcestra["cold_point_height"].values
    orcestra_ct = orcestra["convective_top_height"][-1].values

    print("z / m | Cold point | Convective top")
    print("--- | --- | ---")
    print(f"GATE | {gate_cp:.0f} | {gate_ct:.0f}")
    print(f"ORCESTRA | {orcestra_cp:.0f} | {orcestra_ct:.0f}")
    print(f"Diff |  {orcestra_cp - gate_cp:.0f} | {orcestra_ct - gate_ct:.0f} ")


# -

# GATE
gate_co2 = 303.5e-6
gate = run_rce(co2=gate_co2, o3="gate", sst=299.1)  # 300

# ORCESTRA
orcestra_co2 = 422.8e-6
orcestra_co2_e = (
    np.exp(1.5 * np.log(orcestra_co2 / gate_co2)) * gate_co2
)  # CO2 equivalent forcing
orcestra = run_rce(co2=orcestra_co2_e, o3="orcestra", sst=300.4)  # 301.3


fig, ax = plot_comparison(gate, orcestra)
fig.savefig("gate_vs_orcestra_rce.png")


print_changes(gate, orcestra)
# %%

fig, ax = plot_comparison(gate, orcestra)
ax.plot([-1, -2.5], [21, 23], color="red", lw=2, ls="-", zorder=10)
