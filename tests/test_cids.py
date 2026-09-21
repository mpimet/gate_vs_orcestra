import xarray as xr
import gate_vs_orcestra.utilities.data_utils as data
import pytest
import zarr


@pytest.fixture
def cids():
    return data.get_cids()


def test_open_radiosondes(cids):
    ds = data.open_radiosondes(cids["radiosondes"])
    assert isinstance(ds, xr.Dataset)
    assert "launch_lat" in ds.coords
    assert "launch_lon" in ds.coords
    # assert "altitude" in ds.dims


def test_open_gate(cids):
    ds = data.open_gate(cids["gate"])
    assert isinstance(ds, xr.Dataset)
    assert "launch_lat" in ds.coords
    assert "launch_lon" in ds.coords
    assert "altitude" in ds.dims


def test_open_all(cids):
    failed = []
    for name, cid in cids.items():
        if name == "orcestra":
            continue  # Skip ORCESTRA HEAD CID

        try:
            xr.open_dataset(f"ipfs://{cid}", engine="zarr")
        except zarr.errors.GroupNotFoundError:
            try:
                for id in cid:
                    xr.open_dataset(f"ipfs://{id}", engine="zarr")
            except Exception:
                failed.append(name)

    if failed:
        raise Exception(f"Could not retrieve the following datasets: {failed}")
