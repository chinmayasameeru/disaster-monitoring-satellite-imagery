#!/usr/bin/env python3
"""
USGS Earthquake API Client
Real seismic data from USGS Earthquake Hazards Program.
No authentication required.
API docs: https://earthquake.usgs.gov/fdsnws/event/1/
"""

import httpx
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class USGSEarthquakeClient:
    """Client for USGS Earthquake API (no authentication required)."""

    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def __init__(self, timeout: float = 30.0):
        self.client = httpx.Client(timeout=timeout)

    def query(
        self,
        start_time: str,
        end_time: str,
        min_magnitude: float = 4.0,
        max_magnitude: Optional[float] = None,
        min_latitude: Optional[float] = None,
        max_latitude: Optional[float] = None,
        min_longitude: Optional[float] = None,
        max_longitude: Optional[float] = None,
        max_depth_km: Optional[float] = None,
        limit: int = 200,
        orderby: str = "time",
    ) -> List[Dict]:
        """
        Query earthquake events.

        Args:
            start_time: Start time (ISO 8601 or YYYY-MM-DD)
            end_time: End time (ISO 8601 or YYYY-MM-DD)
            min_magnitude: Minimum magnitude
            max_magnitude: Maximum magnitude
            min_latitude: Minimum latitude
            max_latitude: Maximum latitude
            min_longitude: Minimum longitude
            max_longitude: Maximum longitude
            max_depth_km: Maximum depth in km
            limit: Maximum number of results
            orderby: Sort order (time, magnitude, distance)

        Returns:
            List of earthquake event dicts with 'id', 'time', 'lat', 'lon',
            'depth_km', 'magnitude', 'place', 'url'
        """
        params = {
            "format": "geojson",
            "starttime": start_time,
            "endtime": end_time,
            "minmagnitude": min_magnitude,
            "orderby": orderby,
            "limit": limit,
        }

        if max_magnitude is not None:
            params["maxmagnitude"] = max_magnitude
        if min_latitude is not None:
            params["minlatitude"] = min_latitude
        if max_latitude is not None:
            params["maxlatitude"] = max_latitude
        if min_longitude is not None:
            params["minlongitude"] = min_longitude
        if max_longitude is not None:
            params["maxlongitude"] = max_longitude
        if max_depth_km is not None:
            params["maxdepth"] = max_depth_km

        response = self.client.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        events = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [None, None, None])
            events.append(
                {
                    "id": feature.get("id", ""),
                    "time": props.get("time", ""),
                    "updated": props.get("updated", ""),
                    "lat": coords[1],
                    "lon": coords[0],
                    "depth_km": coords[2] if len(coords) > 2 else None,
                    "magnitude": props.get("mag"),
                    "magnitude_type": props.get("magType"),
                    "place": props.get("place"),
                    "url": props.get("url"),
                    "felt": props.get("felt"),
                    "cdi": props.get("cdi"),  # Community Decimal Intensity
                    "mmi": props.get("mmi"),  # Modified Mercalli Intensity
                    "tsunami": props.get("tsunami", 0) == 1,
                    "alert": props.get("alert"),  # PAGER alert level
                    "status": props.get("status"),
                    "types": props.get("types"),
                }
            )

        return events

    def get_recent_significant(
        self, days: int = 30, min_magnitude: float = 4.5
    ) -> List[Dict]:
        """Get significant earthquakes in the last N days."""
        from datetime import timedelta

        end = datetime.utcnow()
        start = end - timedelta(days=days)
        return self.query(
            start_time=start.strftime("%Y-%m-%d"),
            end_time=end.strftime("%Y-%m-%d"),
            min_magnitude=min_magnitude,
            limit=500,
        )

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
