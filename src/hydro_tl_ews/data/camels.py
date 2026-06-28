"""CAMELS-US dataset loader.

Provides a thin wrapper around the canonical CAMELS-US directory layout used
by NeuralHydrology and Kratzert et al. (2019).  The full dataset (∼3 GB) is
not bundled with this repository — point ``camels_root`` at a local copy or
use the synthetic fallback ``synthetic_camels`` for smoke tests.

CAMELS-US:
    https://ral.ucar.edu/solutions/products/camels
    Catchment attributes:  https://doi.org/10.5065/D6G73C3Q
    Time series:           https://doi.org/10.5065/D6MW2F4D
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

# Forcing variables used by Kratzert et al. (2019)
DYNAMIC_FEATURES = [
    "prcp(mm/day)",      # Precipitation
    "tmax(C)",           # Maximum air temperature
    "tmin(C)",           # Minimum air temperature
    "srad(W/m2)",        # Shortwave radiation
    "vp(Pa)",            # Vapor pressure
    "dayl(s)",           # Day length
]

# Canonical CAMELS attribute groups (Addor et al., 2017).  Subset chosen for
# parity with regional EA-LSTM benchmarks.
STATIC_ATTRIBUTES = [
    # Topography
    "elev_mean", "slope_mean", "area_gages2",
    # Climate
    "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    # Land cover
    "frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff",
    # Soil
    "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity",
    "soil_conductivity", "max_water_content", "sand_frac", "silt_frac",
    "clay_frac",
    # Geology
    "carbonate_rocks_frac", "geol_permeability",
]

TARGET_VARIABLE = "QObs(mm/d)"


@dataclass
class BasinData:
    """Per-basin time series + static attribute tuple."""
    basin_id: str
    forcings: pd.DataFrame  # daily, indexed by date, columns = DYNAMIC_FEATURES
    streamflow: pd.Series   # daily, mm/day
    attributes: pd.Series   # static, indexed by attribute name


class CamelsDataset:
    """Light reader for CAMELS-US daily forcings, streamflow, and attributes.

    Expects the canonical NCAR layout::

        camels_root/
          basin_dataset_public_v1p2/
            basin_mean_forcing/daymet/<huc>/<basin_id>_lump_cida_forcing_leap.txt
            usgs_streamflow/<huc>/<basin_id>_streamflow_qc.txt
          camels_attributes_v2.0/
            camels_topo.txt, camels_clim.txt, camels_soil.txt, ...
    """

    def __init__(self, camels_root: str | Path):
        self.root = Path(camels_root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"CAMELS root not found: {self.root}. "
                f"Download from https://ral.ucar.edu/solutions/products/camels "
                "or use synthetic_camels.SyntheticCamels for smoke tests."
            )
        self._attributes: pd.DataFrame | None = None

    # ------------------------------------------------------------------ attrs
    def load_attributes(self) -> pd.DataFrame:
        if self._attributes is not None:
            return self._attributes
        attr_dir = self.root / "camels_attributes_v2.0"
        files = ["camels_topo.txt", "camels_clim.txt", "camels_hydro.txt",
                 "camels_land.txt", "camels_soil.txt", "camels_geol.txt"]
        frames = []
        for fname in files:
            path = attr_dir / fname
            if not path.exists():
                log.warning("Attribute file missing: %s", path)
                continue
            df = pd.read_csv(path, sep=";", dtype={"gauge_id": str})
            df = df.set_index("gauge_id")
            frames.append(df)
        if not frames:
            raise FileNotFoundError("No CAMELS attribute files found.")
        merged = pd.concat(frames, axis=1)
        # Drop duplicated columns from concat
        merged = merged.loc[:, ~merged.columns.duplicated()]
        self._attributes = merged
        return merged

    # --------------------------------------------------------------- forcings
    def _find_forcing_file(self, basin_id: str) -> Path:
        forcing_dir = self.root / "basin_dataset_public_v1p2" / "basin_mean_forcing" / "daymet"
        for huc_dir in forcing_dir.glob("*"):
            candidate = huc_dir / f"{basin_id}_lump_cida_forcing_leap.txt"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Forcings for basin {basin_id} not found under {forcing_dir}.")

    def _find_streamflow_file(self, basin_id: str) -> Path:
        flow_dir = self.root / "basin_dataset_public_v1p2" / "usgs_streamflow"
        for huc_dir in flow_dir.glob("*"):
            candidate = huc_dir / f"{basin_id}_streamflow_qc.txt"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Streamflow for basin {basin_id} not found under {flow_dir}.")

    def load_forcings(self, basin_id: str) -> pd.DataFrame:
        path = self._find_forcing_file(basin_id)
        # Header: 4 metadata rows then column names
        df = pd.read_csv(path, sep=r"\s+", skiprows=3)
        df["date"] = pd.to_datetime(df[["Year", "Mnth", "Day"]].rename(
            columns={"Year": "year", "Mnth": "month", "Day": "day"}))
        df = df.set_index("date")
        return df[DYNAMIC_FEATURES]

    def load_streamflow(self, basin_id: str, area_km2: float | None = None) -> pd.Series:
        path = self._find_streamflow_file(basin_id)
        df = pd.read_csv(path, sep=r"\s+", header=None,
                         names=["basin", "Year", "Mnth", "Day", "QObs(cfs)", "flag"])
        df["date"] = pd.to_datetime(df[["Year", "Mnth", "Day"]].rename(
            columns={"Year": "year", "Mnth": "month", "Day": "day"}))
        df = df.set_index("date")
        # cfs -> mm/day requires basin area; fall back to attributes
        if area_km2 is None:
            attrs = self.load_attributes()
            area_km2 = float(attrs.loc[basin_id, "area_gages2"])
        cfs_to_mm_per_day = 28316.846592 * 86400 / (area_km2 * 1e6) / 1000.0
        q = df["QObs(cfs)"].replace(-999.0, np.nan) * cfs_to_mm_per_day
        q.name = TARGET_VARIABLE
        return q

    # -------------------------------------------------------------------- io
    def load_basin(self, basin_id: str) -> BasinData:
        attrs_all = self.load_attributes()
        attrs = attrs_all.loc[basin_id, STATIC_ATTRIBUTES]
        forcings = self.load_forcings(basin_id)
        flow = self.load_streamflow(basin_id, area_km2=float(attrs_all.loc[basin_id, "area_gages2"]))
        return BasinData(basin_id=basin_id, forcings=forcings, streamflow=flow, attributes=attrs)

    def load_basins(self, basin_ids: Iterable[str]) -> dict[str, BasinData]:
        return {b: self.load_basin(b) for b in basin_ids}
