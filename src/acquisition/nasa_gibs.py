#!/usr/bin/env python3
"""
NASA GIBS API Client
Real satellite imagery tiles from NASA Global Imagery Browse Services.
No authentication required.
API docs: https://nasa-gibs.github.io/gibs-api-docs/
"""

import httpx
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class NASAGIBSClient:
    """
    Client for NASA GIBS (Global Imagery Browse Services).

    Provides access to near-real-time satellite imagery including:
    - True color (MODIS Terra/Aqua, VIIRS)
    - False color (vegetation, burn scars)
    - NDVI composites
    - Elevation (SRTM, NASADEM)
    - Land cover (MCD12Q1)
    - Nighttime lights (VIIRS)

    No authentication required. Uses JPEG tiles for most layers.
    API docs: https://nasa-gibs.github.io/gibs-api-docs/

    URL format:
    https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/{layer}/default/{date}/{res}/{z}/{y}/{x}.{ext}
    """

    # GIBS WMTS REST tile endpoints (updated 2024 format)
    BASE_URL = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best"
    WEB_MERCATOR_URL = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"

    # Available layers with their supported formats and resolutions
    LAYERS = {
        "true_color": {
            "id": "MODIS_Terra_CorrectedReflectance_TrueColor",
            "resolutions": ["250m", "500m", "1km"],
            "format": "jpg",
        },
        "aqua_true_color": {
            "id": "MODIS_Aqua_CorrectedReflectance_TrueColor",
            "resolutions": ["250m", "500m", "1km"],
            "format": "jpg",
        },
        # Additional layers documented in GIBS API docs — layer IDs
        # must match NASA's published names. See:
        # https://nasa-gibs.github.io/gibs-api-docs/layers/
        "viirs_true_color": {
            "id": "VIIRS_NOAA20_CorrectedReflectance_TrueColor",
            "resolutions": ["250m", "500m", "1km"],
            "format": "jpg",
        },
        # Cloud-minimized annual mosaic — no specific date required
        "blue_marble": {
            "id": "BlueMarble_ShadedRelief",
            "resolutions": ["250m", "500m", "1km"],
            "format": "jpg",
        },
        # For unlisted layers, pass the raw NASA layer name directly
        # via the layer parameter as a string instead of a dict key.
    }

    # Maximum zoom level per projection (EPSG:4326 supports up to 7)
    MAX_ZOOM = {"epsg4326": 7, "epsg3857": 12}

    def __init__(self, timeout: float = 30.0, projection: str = "epsg4326"):
        self.timeout = timeout
        if projection == "epsg3857":
            self.base_url = self.WEB_MERCATOR_URL
            self.max_zoom = self.MAX_ZOOM["epsg3857"]
        else:
            self.base_url = self.BASE_URL
            self.max_zoom = self.MAX_ZOOM["epsg4326"]
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "DisasterMonitor/1.0"},
        )

    def get_tile(
        self,
        layer: str,
        date: str,
        z: int,
        x: int,
        y: int,
        resolution: Optional[str] = None,
    ) -> bytes:
        """
        Fetch a single WMTS tile.

        Args:
            layer: Layer key from LAYERS dict
            date: Date string (YYYY-MM-DD)
            z: Zoom level (0-7 for epsg4326, 0-12 for epsg3857)
            x: Tile column
            y: Tile row
            resolution: Pixel resolution (e.g. "250m"). Auto-selected if None.

        Returns:
            Tile image bytes (JPEG or PNG)
        """
        if layer in self.LAYERS:
            layer_info = self.LAYERS[layer]
            layer_id = layer_info["id"]
            ext = layer_info["format"]
            if resolution is None:
                resolution = layer_info["resolutions"][0]
        else:
            layer_id = layer
            ext = "jpg"
            if resolution is None:
                resolution = "250m"

        z = min(z, self.max_zoom)

        url = (
            f"{self.base_url}/{layer_id}/default/{date}/{resolution}"
            f"/{z}/{y}/{x}.{ext}"
        )

        response = self.client.get(url)
        if response.status_code != 200:
            logger.warning(
                f"GIBS tile fetch failed: {layer} z={z} x={x} y={y} "
                f"-> HTTP {response.status_code}"
            )
            return b""
        return response.content

    def download_tile_range(
        self,
        layer: str,
        date: str,
        bbox: Tuple[float, float, float, float],
        z: int,
        output_dir: Path,
        resolution: Optional[str] = None,
    ) -> List[Path]:
        """
        Download all tiles covering a bounding box.

        Args:
            layer: Layer key from LAYERS dict
            date: Date string (YYYY-MM-DD)
            bbox: (min_lon, min_lat, max_lon, max_lat)
            z: Zoom level (clamped to projection max)
            output_dir: Directory to save tiles
            resolution: Pixel resolution (auto if None)

        Returns:
            List of saved tile file paths
        """
        from math import radians, log, tan, pi, cos

        def deg2num(lat_deg, lon_deg, zoom):
            """Convert lat/lon to EPSG:4326 tile coordinates."""
            lat_rad = radians(lat_deg)
            n = 2.0 ** zoom
            xtile = int((lon_deg + 180.0) / 360.0 * n)
            ytile = int(
                (1.0 - log(tan(lat_rad) + 1.0 / cos(lat_rad)) / pi) / 2.0 * n
            )
            return max(0, xtile), max(0, ytile)

        min_lon, min_lat, max_lon, max_lat = bbox

        # GIBS EPSG:4326 y-axis is inverted relative to WebMercator
        # In GIBS 4326: y=0 is at the north pole
        x_min, y_top = deg2num(max_lat, min_lon, z)
        x_max, y_bottom = deg2num(min_lat, max_lon, z)

        tiles = []
        for x in range(x_min, x_max + 1):
            for y in range(y_top, y_bottom + 1):
                tile_data = self.get_tile(layer, date, z, x, y, resolution)
                if tile_data:
                    ext = self.LAYERS[layer]["format"] if layer in self.LAYERS else "jpg"
                    filename = f"{layer}_{date}_{z}_{x}_{y}.{ext}"
                    filepath = Path(output_dir) / filename
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    with open(filepath, "wb") as f:
                        f.write(tile_data)
                    tiles.append(filepath)
                    logger.debug(f"Saved tile: {filepath}")

        return tiles

    def get_available_dates(self, layer: str) -> List[str]:
        """
        Get available dates for a layer.

        Most GIBS layers are updated daily with near-real-time data.
        """
        if layer not in self.LAYERS:
            return []
        layer_id = self.LAYERS[layer]["id"]
        resolution = self.LAYERS[layer]["resolutions"][0]

        # GIBS DescribeDomains endpoint for date range
        url = f"{self.base_url}/{layer_id}/default/all/{resolution}/all/0.xml"
        try:
            response = self.client.get(url)
            if response.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.text)
                dates = []
                for elem in root.iter():
                    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                    if tag == "Dimension" and elem.text:
                        dates.append(elem.text.strip())
                return dates[:30]
        except Exception as e:
            logger.warning(f"Failed to get dates for {layer}: {e}")
        return []

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
