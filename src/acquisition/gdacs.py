#!/usr/bin/env python3
"""
GDACS API Client
Real global disaster alerts from Global Disaster Alert and Coordination System.
No authentication required.
API docs: https://www.gdacs.org/xml/rss.xml (RSS feed)
"""

import httpx
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class GDACSClient:
    """Client for GDACS RSS feed (no authentication required)."""

    RSS_URL = "https://www.gdacs.org/xml/rss.xml"
    ATOM_URL = "https://www.gdacs.org/xml/rss_7d.xml"  # 7-day feed

    def __init__(self, timeout: float = 30.0):
        self.client = httpx.Client(timeout=timeout)

    def get_current_alerts(self, days: int = 7) -> List[Dict]:
        """
        Get current disaster alerts from GDACS.

        Args:
            days: Number of days to look back (7 or 30)

        Returns:
            List of alert dicts with 'title', 'link', 'description',
            'pub_date', 'alert_level', 'disaster_type', 'country',
            'magnitude', 'lat', 'lon'
        """
        url = self.ATOM_URL if days <= 7 else self.RSS_URL
        response = self.client.get(url)
        response.raise_for_status()
        xml_text = response.text

        root = ET.fromstring(xml_text)
        items = root.findall(".//item")

        alerts = []
        for item in items:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            description = item.findtext("description", "")
            pub_date = item.findtext("pubDate", "")
            guid = item.findtext("guid", "")

            # Parse GDACS-specific elements
            alert_level = ""
            disaster_type = ""
            country = ""
            magnitude = None
            lat = None
            lon = None

            # GDACS uses custom namespace elements
            for child in item:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if tag == "alertlevel":
                    alert_level = child.text or ""
                elif tag == "disastertype":
                    disaster_type = child.text or ""
                elif tag == "country":
                    country = child.text or ""
                elif tag == "magnitude":
                    try:
                        magnitude = float(child.text)
                    except (ValueError, TypeError):
                        pass
                elif tag == "latitude":
                    try:
                        lat = float(child.text)
                    except (ValueError, TypeError):
                        pass
                elif tag == "longitude":
                    try:
                        lon = float(child.text)
                    except (ValueError, TypeError):
                        pass
                elif tag == "eventid":
                    event_id = child.text or ""

            alerts.append(
                {
                    "title": title,
                    "link": link,
                    "description": description,
                    "pub_date": pub_date,
                    "guid": guid,
                    "alert_level": alert_level,
                    "disaster_type": disaster_type,
                    "country": country,
                    "magnitude": magnitude,
                    "lat": lat,
                    "lon": lon,
                }
            )

        return alerts

    def get_alerts_by_type(self, disaster_type: str, days: int = 7) -> List[Dict]:
        """Filter alerts by disaster type (Earthquake, Flood, Cyclone, Drought, etc.)."""
        alerts = self.get_current_alerts(days)
        return [
            a for a in alerts if disaster_type.lower() in a["disaster_type"].lower()
        ]

    def get_alerts_by_country(self, country: str, days: int = 7) -> List[Dict]:
        """Filter alerts by country."""
        alerts = self.get_current_alerts(days)
        return [a for a in alerts if country.lower() in a["country"].lower()]

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
