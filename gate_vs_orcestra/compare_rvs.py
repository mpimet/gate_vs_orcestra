# %%
# - Define some functions for plotting and evaluating atmospheric soundings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import utilities.data_utils as dus
import utilities.preprocessing as pre

# %%
# - load data
#
beach = dus.open_dropsondes(
    "ipfs://bafybeicb33v6ohezyhgq5rumq4g7ejnfqxzcpuzd4i2fxnlos5d7ebmi3m"
)
rapsodi = dus.open_radiosondes(
    "ipfs://bafybeigensqyqxfyaxgyjhwn6ytdpi3i4sxbtffd4oc27zbimyro4hygjq"
)
gate = dus.open_gate("QmSckNEWYkNb1JGVgDUNoQptuE12Czn37WpKpV8pZ3QJiU")

platforms = np.unique(gate.platform_id.values).tolist()
platforms.append("ALL")
# %%
# - plot gate data
#
sns.set_context(context="paper")

for RV in platforms:
    fig, ax = plt.subplots(1, 5, figsize=(5, 3), sharey=True)
    i = 0
    if RV == "ALL":
        ds = gate.pipe(pre.sel_gate_A)
    else:
        ds = gate.isel(sonde=(gate.platform_id == RV).compute()).pipe(pre.sel_gate_A)
    for fld in ["ta", "rh", "u", "v", "p"]:
        x = ds[fld].stack(points=["sonde", "altitude"])
        ax[i].scatter(x, x.altitude, s=1, color="k", alpha=0.003)
        ds[fld].mean(dim="sonde").plot(ax=ax[i], y="altitude", lw=1.0, color="w")

        ax[i].set_ylim(0, 32000)
        ax[i].set_ylabel(None)
        i += 1

    ax[0].set_ylabel("altitude / km")
    ax[0].set_ylabel("altitude / km")
    ax[0].set_yticks(np.arange(0, 32000, 3000))
    ax[0].set_yticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
    ax[0].set_xlabel("$T$ / K")
    ax[1].set_xlabel("$RH$ / -")
    ax[2].set_xlabel("$u$ / m s$^{-1}$")
    ax[3].set_xlabel("$v$ / m s$^{-1}$")
    ax[4].set_xlabel("$P$ / Pa")

    sns.despine(offset=5)
    plt.suptitle(RV)
    plt.savefig(f"plots/{RV}_outliers.png", dpi=600)
# %%
