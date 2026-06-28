"""USGS NWIS streamflow retrieval helpers.

Wraps the official ``dataretrieval`` Python package (USGS).  Designed to run
only when an internet connection is available; the synthetic generator covers
the offline smoke-test path.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

CFS_TO_M3S = 0.0283168466


def fetch_daily_streamflow(
    site_no: str,
    start: str = "1980-01-01",
    end: Optional[str] = None,
    parameter_cd: str = "00060",  # discharge in cfs
) -> pd.DataFrame:
    """Download daily mean streamflow from USGS NWIS for a single site.

    Returns a DataFrame indexed by date with columns ``q_cfs`` and ``q_m3s``.
    """
    try:
        from dataretrieval import nwis
    except ImportError as e:
        raise ImportError(
            "dataretrieval is required for live USGS NWIS pulls. "
            "Install with `pip install dataretrieval`."
        ) from e
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    log.info("Fetching USGS NWIS site %s (%s -> %s)", site_no, start, end)
    df, _ = nwis.get_dv(sites=site_no, parameterCd=parameter_cd,
                        start=start, end=end)
    if df.empty:
        return pd.DataFrame(columns=["q_cfs", "q_m3s"])
    # Column varies by site; pick the first parameter column
    param_col = [c for c in df.columns if c.startswith(parameter_cd)][0]
    out = pd.DataFrame({"q_cfs": pd.to_numeric(df[param_col], errors="coerce")})
    out["q_m3s"] = out["q_cfs"] * CFS_TO_M3S
    out.index.name = "date"
    return out


def cfs_to_mm_per_day(q_cfs: pd.Series, area_km2: float) -> pd.Series:
    """Convert discharge (cfs) to specific runoff (mm/day) given basin area."""
    cfs_to_mm = 28316.846592 * 86400 / (area_km2 * 1e6) / 1000.0
    return q_cfs * cfs_to_mm
