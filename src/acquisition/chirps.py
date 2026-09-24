#!/usr/bin/env python3
"""
CHIRPS Precipitation Data Client
Real precipitation data from Climate Hazards Group InfraRed Precipitation with Station data.
No authentication required.
Data source: https://www.chc.ucsb.edu/data/chirps
"""

import httpx
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import gzip

logger = logging.getLogger(__name__)


class CHIRPSClient:
    """Client for CHIRPS precipitation data (no authentication required)."""

    BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0"

    def __init__(self, timeout: float = 60.0):
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    def get_daily_url(self, date: str, region: str = "global") -> str:
        """
        Get CHIRPS daily data URL.

        Args:
            date: Date string (YYYY-MM-DD)
            region: Region (global, africa, west_africa)

        Returns:
            URL to the .tif.gz file
        """
        dt = datetime.strptime(date, "%Y-%m-%d")
        if region == "global":
            return (
                f"{self.BASE_URL}/global_daily/tifs/p05/{dt.year}/"
                f"chirps-v2.0.{dt.year}.{dt.month:02d}.{dt.day:02d}.tif.gz"
            )
        elif region == "africa":
            return (
                f"{self.BASE_URL}/africa_daily/tifs/p05/{dt.year}/"
                f"chirps-v2.0.{dt.year}.{dt.month:02d}.{dt.day:02d}.tif.gz"
            )
        raise ValueError(f"Unsupported region: {region}")

    def get_monthly_url(self, year: int, month: int, region: str = "global") -> str:
        """Get CHIRPS monthly data URL."""
        if region == "global":
            return f"{self.BASE_URL}/global_monthly/tifs/chirps-v2.0.{year}.{month:02d}.tif"
        elif region == "africa":
            return f"{self.BASE_URL}/africa_monthly/tifs/chirps-v2.0.{year}.{month:02d}.tif"
        raise ValueError(f"Unsupported region: {region}")

    def download_daily(
        self, date: str, output_dir: Path, region: str = "global"
    ) -> Optional[Path]:
        """
        Download daily CHIRPS precipitation data.

        Returns:
            Path to downloaded .tif file, or None if download failed
        """
        url = self.get_daily_url(date, region)
        filename = url.split("/")[-1].replace(".gz", "")
        filepath = Path(output_dir) / filename

        if filepath.exists():
            return filepath

        logger.info(f"Downloading CHIRPS daily: {url}")
        response = self.client.get(url)
        if response.status_code != 200:
            logger.warning(f"CHIRPS daily download failed: HTTP {response.status_code}")
            return None

        gz_path = filepath.with_suffix(".tif.gz")
        with open(gz_path, "wb") as f:
            f.write(response.content)

        # Decompress gzip
        with gzip.open(gz_path, "rb") as f_in:
            with open(filepath, "wb") as f_out:
                f_out.write(f_in.read())
        gz_path.unlink()

        logger.info(f"CHIRPS daily saved: {filepath}")
        return filepath

    def download_monthly(
        self, year: int, month: int, output_dir: Path, region: str = "global"
    ) -> Optional[Path]:
        """Download monthly CHIRPS precipitation data."""
        url = self.get_monthly_url(year, month, region)
        filename = url.split("/")[-1]
        filepath = Path(output_dir) / filename

        if filepath.exists():
            return filepath

        logger.info(f"Downloading CHIRPS monthly: {url}")
        response = self.client.get(url)
        if response.status_code != 200:
            logger.warning(
                f"CHIRPS monthly download failed: HTTP {response.status_code}"
            )
            return None

        with open(filepath, "wb") as f:
            f.write(response.content)

        logger.info(f"CHIRPS monthly saved: {filepath}")
        return filepath

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
