import xarray as xr
import gate_vs_orcestra.utilities.data_utils as data
import pytest


@pytest.fixture
def cids():
    return data.get_cids()


def test_open_dropsondes(cids):
    ds = data.open_dropsondes(cids["dropsondes"])
    assert isinstance(ds, xr.Dataset)
    assert "launch_lat" in ds.coords
    assert "launch_lon" in ds.coords
    assert "altitude" in ds.dims


def test_open_radiosondes(cids):
    ds = data.open_radiosondes(cids["radiosondes"])
    assert isinstance(ds, xr.Dataset)
    assert "launch_lat" in ds.coords
    assert "launch_lon" in ds.coords
    assert "altitude" in ds.dims


def test_open_gate(cids):
    ds = data.open_gate(cids["gate"])
    assert isinstance(ds, xr.Dataset)
    assert "launch_lat" in ds.coords
    assert "launch_lon" in ds.coords
    assert "altitude" in ds.dims
    assert "sonde_id" in ds.coords
