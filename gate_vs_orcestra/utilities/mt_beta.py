import numpy as np
import moist_thermodynamics.functions as mt
import moist_thermodynamics.saturation_vapor_pressures as svp
import moist_thermodynamics.constants as constants
from scipy.integrate import solve_ivp


def moist_adiabat(
    Tbeg,
    Pbeg,
    Pend,
    dP,
    qt,
    cc=constants.cl,
    lv=mt.vaporization_enthalpy,
    es=svp.es_default,
):
    """Returns the temperature and pressure by integrating along a moist adiabat

    Deriving the moist adiabats by assuming a constant moist potential temperature
    provides a Rankine-Kirchoff approximation to the moist adiabat.  If thermodynamic
    constants are allowed to vary with temperature then the intergation must be
    performed numerically, as outlined here for the case of constant thermodynamic
    constants and no accounting for the emergence of a solid condensage phase (ice).

    The introduction of this function allows one to estimate, for instance, the effect of
    isentropic freezing on the moist adiabat as follows:

    Tliq,Px= moist_adiabat(Tsfc,Psfc,Ptop,dP,qt,cc=constants.cl,lv=mt.vaporization_enthalpy,es = mt.es_mxd)
    Tice,Py= moist_adiabat(Tsfc,Psfc,Ptop,dP,qt,cc=constants.ci,lv=mt.sublimation_enthalpy ,es = mt.es_mxd)

    T  = np.ones(len(Tx))*constants.T0
    T[Tliq>constants.T0] = Tliq[Tliq>constants.T0]
    T[Tice<constants.T0] = Tice[Tice<constants.T0]

    which introduces an isothermal layer in the region where the fusion enthalpy is sufficient to do
    the expansional work

    Args:
        Tbeg: temperature at P0 in kelvin
        Pbeg: starting pressure in pascal
        Pend: pressure to which to integrate to in pascal
        dP:   integration step
        qt:   specific mass of total water
        es:   saturation vapor expression

    """
    Tbeg = np.asarray(Tbeg).reshape(1)
    Pbeg = np.asarray(Pbeg).reshape(1)[0]
    Pend = np.asarray(Pend).reshape(1)[0]
    dP = np.asarray(dP).reshape(1)[0]

    def f(P, T, qt, cc, lv):
        Rd = constants.Rd
        Rv = constants.Rv
        cpd = constants.cpd
        cpv = constants.cpv

        qv = mt.saturation_partition(P, es(T), qt)
        qc = qt - qv
        qd = 1.0 - qt

        R = qd * Rd + qv * Rv
        cp = qd * cpd + qv * cpv + qc * cc
        vol = R * T / P

        dX_dT = cp
        dX_dP = vol
        if qc > 0.0:
            beta_P = R / (qd * Rd)
            beta_T = beta_P * lv(T) / (Rv * T)

            dX_dT += lv(T) * qv * beta_T / T
            dX_dP *= 1.0 + lv(T) * qv * beta_P / (R * T)
        return dX_dP / dX_dT

    r = solve_ivp(
        f,
        [Pbeg, Pend],
        y0=Tbeg,
        args=(qt, cc, lv),
        t_eval=np.arange(Pbeg, Pend, -dP),
        method="LSODA",
        rtol=1.0e-5,
        atol=1.0e-8,
    )
    return r.y[0], r.t


def get_adiabat(P, Tsfc=301.0, qsfc=17e-3, Tmin=200.0, thx=mt.theta_l, integrate=False):
    """Returns the moist adiabat along a pressure dimension.

    Cacluates the moist adiabate based either on an integration or a specified
    isentrope with pressure as the vertical coordinate.

    Args:
        P: pressure
        Tsfc: starting (value at P.max()) temperature
        qsfc: starting (value at P.max()) specific humidity
        Tmin: minimum temperature of adiabat
        thx: function to calculate isentrope if integrate = False
        integrate: determines if explicit integration will be used.
    """

    es = svp.liq_analytic
    T0 = constants.T0
    i4T = np.vectorize(mt.invert_for_temperature)

    Tx = thx(Tsfc, P[0], qsfc)
    TK = i4T(thx, Tx, P, qsfc)

    if integrate:
        es = mt.make_es_mxd(es_liq=svp.liq_analytic, es_ice=svp.ice_analytic)
        dP = P[1] - P[0]
        Ptop = P[-1] + dP
        Tice, Py = moist_adiabat(
            Tsfc,
            P[0],
            Ptop,
            -dP,
            qsfc,
            cc=constants.ci,
            lv=mt.sublimation_enthalpy,
            es=es,
        )
        Tliq, Px = moist_adiabat(
            Tsfc,
            P[0],
            Ptop,
            -dP,
            qsfc,
            cc=constants.cl,
            lv=mt.vaporization_enthalpy,
            es=es,
        )
        TK = np.ones(len(Px)) * T0
        TK[Tliq > T0] = Tliq[Tliq > T0]
        TK[Tice < T0] = Tice[Tice < T0]

    return np.maximum(TK, Tmin)


def mk_sounding_ds(P, T, q, thx=mt.theta_l, integrate=False):
    import xarray as xr

    TPq = xr.Dataset(
        data_vars={
            "P": (
                ("altitude",),
                P,
                {"units": "Pa", "long_name": "air pressure", "symbol": "$P$"},
            ),
            "T": (
                ("altitude",),
                np.full(len(P), np.nan),
                {"units": "K", "long_name": "Temperature", "symbol": "$T$"},
            ),
            "q": (
                ("altitude",),
                np.full(len(P), 0.0),
                {"units": "Pa", "long_name": "air pressure", "symbol": "$P$"},
            ),
            "altitude": (
                ("altitude",),
                np.full(len(P), np.nan),
                {"units": "m", "long_name": "pressure altitude", "symbol": "$z$"},
            ),
            "theta": (
                ("altitude",),
                np.full(len(P), np.nan),
                {"units": "K", "long_name": "Temperature", "symbol": "$\\theta$"},
            ),
            "rh": (
                ("altitude",),
                np.full(len(P), np.nan),
                {"units": "-", "long_name": "relative humidity", "symbol": "$W$"},
            ),
        }
    )
    TPq["T"] = TPq["T"].copy(
        data=get_adiabat(P, Tsfc=T, qsfc=q, thx=thx, integrate=integrate)
    )
    TPq["q"] = TPq["P"].copy(data=q * np.ones(len(P)))
    TPq["altitude"] = TPq["theta"].copy(
        data=mt.hydrostatic_altitude_np(P, np.asarray(TPq["T"]), np.asarray(TPq["q"]))
    )
    TPq["theta"] = TPq["theta"].copy(data=mt.theta(TPq["T"], P))

    return TPq.set_coords("altitude")
