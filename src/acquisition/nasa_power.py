#!/usr/bin/env python3
"""
NASA POWER API Client
Real meteorological data from NASA Prediction Of Worldwide Energy Resources.
No authentication required.
API docs: https://power.larc.nasa.gov/api/pages/
"""

import httpx
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class NASAPOWERClient:
    """Client for NASA POWER API (no authentication required)."""

    BASE_URL = "https://power.larc.nasa.gov/api/temporal"

    def __init__(self, timeout: float = 30.0):
        self.client = httpx.Client(timeout=timeout)

    def fetch_daily(
        self,
        lon: float,
        lat: float,
        start_date: str,
        end_date: str,
        parameters: Optional[List[str]] = None,
        community: str = "SB",
    ) -> Dict:
        """
        Fetch daily meteorological data for a point location.

        Args:
            lon: Longitude
            lat: Latitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            parameters: List of parameter codes (default: precipitation, temp, wind, RH, solar)
            community: Science community (SB = Sustainable Buildings, AG = Agroclimatology)

        Returns:
            dict with 'properties' containing parameter time series
        """
        if parameters is None:
            parameters = [
                "PRECTOTCORR",  # Precipitation (mm/day)
                "T2M",  # Temperature at 2m (°C)
                "T2M_MAX",  # Max temperature (°C)
                "T2M_MIN",  # Min temperature (°C)
                "WS2M",  # Wind speed at 2m (m/s)
                "RH2M",  # Relative humidity at 2m (%)
                "ALLSKY_SFC_SW_DWN",  # All-sky shortwave downward (MJ/m²/day)
                "PS",  # Surface pressure (kPa)
            ]

        params = {
            "parameters": ",".join(parameters),
            "community": community,
            "longitude": lon,
            "latitude": lat,
            "start": start_date.replace("-", ""),
            "end": end_date.replace("-", ""),
            "format": "JSON",
        }

        response = self.client.get(f"{self.BASE_URL}/daily/point", params=params)
        response.raise_for_status()
        return response.json()

    def fetch_monthly(
        self,
        lon: float,
        lat: float,
        start_date: str,
        end_date: str,
        parameters: Optional[List[str]] = None,
    ) -> Dict:
        """Fetch monthly aggregated meteorological data."""
        if parameters is None:
            parameters = ["PRECTOTCORR", "T2M", "WS2M"]

        params = {
            "parameters": ",".join(parameters),
            "community": "SB",
            "longitude": lon,
            "latitude": lat,
            "start": start_date.replace("-", ""),
            "end": end_date.replace("-", ""),
            "format": "JSON",
        }

        response = self.client.get(f"{self.BASE_URL}/monthly/point", params=params)
        response.raise_for_status()
        return response.json()

    def fetch_climatology(
        self,
        lon: float,
        lat: float,
        parameters: Optional[List[str]] = None,
    ) -> Dict:
        """Fetch long-term climatology (1981-2020 for daily, 1981-2020 for monthly)."""
        if parameters is None:
            parameters = ["PRECTOTCORR", "T2M"]

        params = {
            "parameters": ",".join(parameters),
            "community": "SB",
            "longitude": lon,
            "latitude": lat,
            "start": "19810101",
            "end": "20201231",
            "format": "JSON",
        }

        response = self.client.get(f"{self.BASE_URL}/climatology/point", params=params)
        response.raise_for_status()
        return response.json()

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
