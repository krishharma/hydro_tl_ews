"""Synthetic CAMELS-like generator for smoke tests and CI.

Produces N basins with realistic-shape daily forcings (precip, tmax, tmin,
srad, vp, dayl) and physically plausible streamflow generated from a simple
two-bucket conceptual model whose parameters depend on the basin's static
attributes.  The output structure mirrors :class:`hydro_tl_ews.data.camels.BasinData`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from .camels import (
    BasinData,
    DYNAMIC_FEATURES,
    STATIC_ATTRIBUTES,
    TARGET_VARIABLE,
)


def _seasonal(t: np.ndarray, mean: float, amp: float, phase: float) -> np.ndarray:
    """Sine seasonal cycle."""
    return mean + amp * np.sin(2 * np.pi * (t / 365.25) + phase)


def _generate_forcings(n_days: int, rng: np.random.Generator,
                       base_temp: float, precip_mean: float,
                       snow_basin: bool) -> pd.DataFrame:
    t = np.arange(n_days)
    # Air temperature: seasonal swing larger for snow basins
    swing = 18.0 if snow_basin else 10.0
    tmean = _seasonal(t, mean=base_temp, amp=swing, phase=-np.pi / 2)
    diurnal = rng.normal(0, 2, size=n_days)
    tmax = tmean + 5 + diurnal
    tmin = tmean - 5 + diurnal
    # Precipitation: gamma-distributed with seasonal weighting
    season_w = 1.0 + 0.5 * np.sin(2 * np.pi * t / 365.25 + np.pi)
    prcp = rng.gamma(shape=0.6, scale=precip_mean * season_w / 0.6)
    # Wet-day fraction ~ 0.4
    prcp = np.where(rng.random(n_days) < 0.4, prcp, 0.0)
    # Shortwave radiation: seasonal
    srad = _seasonal(t, mean=200, amp=120, phase=-np.pi / 2) + rng.normal(0, 20, n_days)
    srad = np.clip(srad, 30, None)
    # Vapor pressure: depends on temperature (Pa)
    vp = 611.0 * np.exp((17.27 * tmean) / (tmean + 237.3)) * (0.4 + 0.4 * rng.random(n_days))
    # Day length (s)
    dayl = (43200 + 14400 * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)).astype(float)

    dates = pd.date_range("1990-01-01", periods=n_days, freq="D")
    df = pd.DataFrame(
        {
            "prcp(mm/day)": prcp,
            "tmax(C)": tmax,
            "tmin(C)": tmin,
            "srad(W/m2)": srad,
            "vp(Pa)": vp,
            "dayl(s)": dayl,
        },
        index=dates,
    )
    df = df[DYNAMIC_FEATURES]
    return df


def _two_bucket_streamflow(forcings: pd.DataFrame, attrs: dict[str, float],
                           rng: np.random.Generator) -> pd.Series:
    """Toy two-bucket snow + soil-moisture model -> daily streamflow (mm/day)."""
    prcp = forcings["prcp(mm/day)"].values
    tmean = (forcings["tmax(C)"].values + forcings["tmin(C)"].values) / 2.0
    n = len(prcp)

    # Parameters loosely tied to attributes
    melt_factor = 3.0 + 4.0 * attrs["frac_snow"]                 # mm/°C/day
    storage_capacity = 50.0 + 200.0 * attrs["soil_porosity"]     # mm
    recession = 0.92 - 0.02 * attrs["aridity"]
    et_max = 1.0 + 3.0 * attrs["pet_mean"] / 5.0                  # mm/day cap

    snowpack = 0.0
    soil = storage_capacity * 0.5
    q = np.zeros(n)
    for i in range(n):
        if tmean[i] < 0:
            snowpack += prcp[i]
            liquid_in = 0.0
        else:
            melt = min(snowpack, max(0.0, melt_factor * tmean[i]))
            snowpack -= melt
            liquid_in = prcp[i] + melt
        # Saturation excess
        soil += liquid_in
        excess = max(0.0, soil - storage_capacity)
        soil -= excess
        # ET
        et = min(soil, et_max * max(0.0, tmean[i]) / 20.0)
        soil -= et
        # Baseflow
        baseflow = (1.0 - recession) * soil
        soil -= baseflow
        q[i] = excess + baseflow

    q += np.abs(rng.normal(0, 0.05, n)) * q.mean()  # small obs noise
    return pd.Series(q, index=forcings.index, name=TARGET_VARIABLE)


def _sample_attributes(rng: np.random.Generator, snow_basin: bool) -> dict[str, float]:
    base = {a: 0.0 for a in STATIC_ATTRIBUTES}
    base.update(
        {
            "elev_mean": rng.uniform(2200, 3200) if snow_basin else rng.uniform(100, 1500),
            "slope_mean": rng.uniform(50, 200) if snow_basin else rng.uniform(5, 80),
            "area_gages2": rng.uniform(50, 1500),
            "p_mean": rng.uniform(2, 5),
            "pet_mean": rng.uniform(1.5, 4.0),
            "p_seasonality": rng.uniform(-0.5, 0.5),
            "frac_snow": rng.uniform(0.4, 0.8) if snow_basin else rng.uniform(0.0, 0.2),
            "aridity": rng.uniform(0.5, 1.5),
            "high_prec_freq": rng.uniform(5, 25),
            "high_prec_dur": rng.uniform(1, 4),
            "low_prec_freq": rng.uniform(100, 250),
            "low_prec_dur": rng.uniform(2, 10),
            "frac_forest": rng.uniform(0.2, 0.9),
            "lai_max": rng.uniform(1, 7),
            "lai_diff": rng.uniform(0.5, 5),
            "gvf_max": rng.uniform(0.4, 0.95),
            "gvf_diff": rng.uniform(0.1, 0.6),
            "soil_depth_pelletier": rng.uniform(1, 25),
            "soil_depth_statsgo": rng.uniform(1, 2),
            "soil_porosity": rng.uniform(0.3, 0.55),
            "soil_conductivity": rng.uniform(1, 50),
            "max_water_content": rng.uniform(0.1, 0.5),
            "sand_frac": rng.uniform(20, 80),
            "silt_frac": rng.uniform(10, 60),
            "clay_frac": rng.uniform(5, 40),
            "carbonate_rocks_frac": rng.uniform(0, 0.5),
            "geol_permeability": rng.uniform(-15, -10),
        }
    )
    return {k: base[k] for k in STATIC_ATTRIBUTES}


class SyntheticCamels:
    """Drop-in replacement for :class:`CamelsDataset` for smoke tests."""

    def __init__(self, n_basins: int = 12, n_days: int = 365 * 10,
                 snow_fraction: float = 0.4, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_days = n_days
        self.basins: dict[str, BasinData] = {}
        attr_rows = {}
        for i in range(n_basins):
            basin_id = f"S{i:08d}"
            snow = self.rng.random() < snow_fraction
            attrs = _sample_attributes(self.rng, snow_basin=snow)
            base_temp = 4.0 if snow else 14.0
            forcings = _generate_forcings(
                n_days, self.rng,
                base_temp=base_temp,
                precip_mean=attrs["p_mean"],
                snow_basin=snow,
            )
            flow = _two_bucket_streamflow(forcings, attrs, self.rng)
            self.basins[basin_id] = BasinData(
                basin_id=basin_id,
                forcings=forcings,
                streamflow=flow,
                attributes=pd.Series(attrs),
            )
            attr_rows[basin_id] = attrs
        self._attribute_frame = pd.DataFrame.from_dict(attr_rows, orient="index")

    def load_attributes(self) -> pd.DataFrame:
        return self._attribute_frame

    def load_basin(self, basin_id: str) -> BasinData:
        return self.basins[basin_id]

    def load_basins(self, basin_ids: Iterable[str]) -> dict[str, BasinData]:
        return {b: self.basins[b] for b in basin_ids}

    @property
    def basin_ids(self) -> list[str]:
        return list(self.basins.keys())
