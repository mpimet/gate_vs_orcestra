# %%
# - import modules and define functions

import utilities.data_utils as dus
from utilities.gate_process import mask_outliers, mask_unphysical, fill_gaps, coverage


# %%
# - clean and fill for l3 data
#
attr_dict = dus.variable_attribute_dict

gate_l2_cid = "QmT6psBncPq1ya6hXzmqbA4zA35BBUCwGfnN7ZBFt3jcQC"
gate_l2 = dus.open_gate(gate_l2_cid)

gate_l3 = (
    gate_l2.pipe(mask_unphysical)
    .pipe(fill_gaps, max_igap=1000, max_egap=300)
    .pipe(mask_outliers)
    .sel(altitude=slice(-10, 25000))
    .assign_attrs(
        {
            "title": "GATE radiosonde dataset (Level 3)",
            "summary": "GATE ship radiosondes (filled and cleaned) subsetted to ORCESTRA time of year and lower 25 km",
            "creator_name": "Bjorn Stevens",
            "creator_email": "bjorn.stevens@mpimet.mpg.de",
            "license": "CC-BY-4.0",
            "processing_level": "3",
            "institution": "Max Planck Institute for Meteorologie",
            "source": "radiosonde",
            "history": f"Processed Gate level 2 radiosonde data with {gate_l2_cid} cid",
        }
    )
)

for var, attrs in attr_dict.items():
    gate_l3[var].attrs = attrs

print(
    f"GATE level 3 ta data coverage:\n "
    f"initially = {coverage(gate_l2.ta): .8f}%\n "
    f"finally = {coverage(gate_l3.ta):.8f}%\n "
)

gate_l3.to_zarr("~/data/gate-l3.zarr")
