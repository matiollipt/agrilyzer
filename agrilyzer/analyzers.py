import os
import re
import json
import hashlib
import requests
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Union, Tuple
from agrilyzer.config import cfg


class Agrilyzer:
    """
    Client for NASA POWER daily weather data with persistent on‑disk caching.

    The client will always attempt to load data from the local cache first
    (if enabled) before making a network request, significantly reducing
    bandwidth and improving latency.

    A cache file is uniquely identified by: location, date range, and the
    set of parameters requested.
    """

    BASE_URL = cfg.get(
        "base_url", "https://power.larc.nasa.gov/api/temporal/daily/point"
    )
    PARAMS = cfg.get("params", {}).get("default", {})
    PARAMS_DESCRIPTIONS = cfg.get("params_descriptions", {})

    def __init__(
        self,
        name: str,
        lat: float,
        lon: float,
        start: Union[str, datetime],
        end: Union[str, datetime],
        params: Optional[List[str]] = None,
        params_desc: Optional[Dict[str, str]] = None,
        cache_config: Optional[Dict[str, Union[str, bool]]] = None,
        session: Optional[requests.Session] = None,
    ):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)

        # If the user passes a mapping of param->description, keep order;
        # otherwise use list of codes
        self.params: List[str] = (
            params if params is not None else list(self.PARAMS.keys())
        )
        self.params_desc = params_desc or self.PARAMS_DESCRIPTIONS

        # --- Caching -----------------------------------------------------
        default_cache_cfg = {
            "enabled": True,
            "path": os.path.expanduser("~/.agrilyzer_cache"),
        }
        self.cache_cfg = {**default_cache_cfg, **(cache_config or {})}
        if self.cache_cfg["enabled"]:
            os.makedirs(self.cache_cfg["path"], exist_ok=True)

        # HTTP session
        self.session = session or requests.Session()

        # DataFrames kept in‑memory (lazy loaded)
        self._raw: Optional[Dict[str, Dict[str, float]]] = None
        self._df: Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _cache_filename(self) -> str:
        """Generate a unique filename for this query."""
        loc = re.sub(r"\s+", "_", self.name.lower())
        start = self.start.strftime("%Y%m%d")
        end = self.end.strftime("%Y%m%d")
        # A short hash of the parameter list keeps filenames manageable
        params_hash = hashlib.md5(",".join(sorted(self.params)).encode()).hexdigest()[
            :6
        ]
        return f"{loc}_{start}_{end}_{params_hash}.json"

    def _cache_path(self) -> str:
        return os.path.join(self.cache_cfg["path"], self._cache_filename())

    # ---------------------------------------------------------------------
    # Caching (load / save)
    # ---------------------------------------------------------------------
    def _load_cached(self) -> Optional[Dict[str, Dict[str, float]]]:
        if not self.cache_cfg.get("enabled", False):
            return None
        try:
            with open(self._cache_path(), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def _save_cache(self, raw: Dict[str, Dict[str, float]]) -> None:
        if not self.cache_cfg.get("enabled", False):
            return
        try:
            with open(self._cache_path(), "w") as f:
                json.dump(raw, f)
        except OSError as exc:
            # Non‑fatal: caching failure should not break the workflow
            print(f"[Agrilizer] Warning: could not save cache: {exc}")

    # ---------------------------------------------------------------------
    # Networking
    # ---------------------------------------------------------------------
    def _build_url(self) -> str:
        pstr = ",".join(self.params)
        return (
            f"{self.BASE_URL}?parameters={pstr}"
            f"&community=RE&latitude={self.lat}&longitude={self.lon}"
            f"&start={self.start:%Y%m%d}&end={self.end:%Y%m%d}&format=JSON"
        )

    def _fetch(self, prefer_cache: bool = True) -> Dict[str, Dict[str, float]]:
        """Return raw data dict, consulting cache when possible."""
        if prefer_cache:
            cached = self._load_cached()
            if cached is not None:
                return cached

        # Cache miss → request data
        resp = self.session.get(self._build_url(), timeout=30)
        resp.raise_for_status()

        data = resp.json().get("properties", {}).get("parameter", {})
        if not data:
            raise ValueError("No data returned from NASA POWER API.")

        self._save_cache(data)
        return data

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def _to_df(self, raw: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        df = pd.DataFrame(raw)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index.name = "date"
        return df.reset_index()

    def load_df(self, prefer_cache: bool = True) -> pd.DataFrame:
        """Load data as DataFrame, optionally preferring cache."""
        if self._df is None:
            self._raw = self._fetch(prefer_cache)
            self._df = self._to_df(self._raw)
        return self._df.copy()

    def reload(self, ignore_cache: bool = False) -> pd.DataFrame:
        """Force re‑loading data, bypassing in‑memory copy.

        If *ignore_cache* is True the cache on disk is ignored and a fresh
        request is performed (use sparingly).
        """
        self._raw = self._fetch(prefer_cache=not ignore_cache)
        self._df = self._to_df(self._raw)
        return self._df.copy()

    def get(self, long_names: bool = False, prefer_cache: bool = True) -> pd.DataFrame:
        df = self.load_df(prefer_cache)
        return df.rename(columns=self._col_name_map()) if long_names else df

    def _col_name_map(self) -> Dict[str, str]:
        return {k: self.PARAMS.get(k, k) for k in self.params}

    # ---------------------------------------------------------------------
    # Data wrangling helpers
    # ---------------------------------------------------------------------
    def filter(
        self,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
        prefer_cache: bool = True,
    ) -> pd.DataFrame:
        df = self.get(long_names=True, prefer_cache=prefer_cache)
        if start or end:
            s = pd.to_datetime(start) if start else df.date.min()
            e = pd.to_datetime(end) if end else df.date.max()
            df = df[df.date.between(s, e)]
        if thresholds:
            for col, (lo, hi) in thresholds.items():
                df = df[df[col].between(lo, hi)]
        return df

    def aggregate(
        self,
        freq: str = "M",
        agg: Union[str, Dict[str, Union[str, List[str]]]] = "mean",
        prefer_cache: bool = True,
    ) -> pd.DataFrame:
        return (
            self.get(long_names=True, prefer_cache=prefer_cache)
            .set_index("date")
            .resample(freq)
            .agg(agg)
            .reset_index()
        )

    # ---------------------------------------------------------------------
    # Debug / utility
    # ---------------------------------------------------------------------
    def cache_info(self) -> str:
        """Return path to current cache file and whether it exists."""
        path = self._cache_path()
        return f"Cache {'exists' if os.path.exists(path) else 'missing'} → {path}"
