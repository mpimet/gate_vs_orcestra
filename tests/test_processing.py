import xarray as xr
from gate_vs_orcestra.utilities.preprocessing import (
    sel_sub_domain,
    sel_gate_A,
    sel_percusion_E,
)

ds = xr.Dataset(
    {
        "launch_lon": (["sonde"], [-20, -25, -30, -22]),
        "launch_lat": (["sonde"], [0, 4, 10, 8]),
    },
)


def test_sel_sub_domain():
    polygon = [[-27.0, 0.0], [-23.0, 0.0], [-20.0, 10.0], [-30.0, 9.0]]
    result = sel_sub_domain(ds, polygon)
    expected = xr.Dataset(
        {
            "launch_lon": (["sonde"], [-25, -22.0]),
            "launch_lat": (["sonde"], [4, 8.0]),
        },
    )
    xr.testing.assert_equal(result, expected)


def test_sel_gate_A():
    result = sel_gate_A(ds)
    expected = xr.Dataset(
        {
            "launch_lon": (["sonde"], [-22]),
            "launch_lat": (["sonde"], [8]),
        },
    )
    xr.testing.assert_equal(result, expected)


def test_sel_percusion_E():
    result = sel_percusion_E(ds)
    expected = xr.Dataset(
        {
            "launch_lon": (
                ["sonde"],
                [
                    -25.0,
                    -30,
                    -22,
                ],
            ),
            "launch_lat": (["sonde"], [4, 10, 8]),
        },
    )
    xr.testing.assert_equal(result, expected)
