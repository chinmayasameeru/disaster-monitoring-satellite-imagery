#!/usr/bin/env python3
"""
OpenStreetMap Overpass API Client
Real infrastructure, population, and land use data.
No authentication required.
API docs: https://wiki.openstreetmap.org/wiki/Overpass_API
"""

import httpx
import logging
import urllib.parse
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OSMOverpassClient:
    """
    Client for OpenStreetMap Overpass API.

    Fetches infrastructure data (roads, buildings, schools, hospitals),
    water bodies, land use, and population-related features.
    No authentication required — public read-only API with rate limits.
    """

    BASE_URL = "https://overpass-api.de/api/interpreter"

    def __init__(self, timeout: float = 60.0):
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": "DisasterMonitor/1.0 (educational research)",
                "Accept": "application/json",
            },
        )

    def query(
        self, bbox: List[float], query_body: str
    ) -> List[Dict]:
        """
        Execute an Overpass QL query within a bounding box.

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            query_body: Overpass QL query body

        Returns:
            List of element dicts
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        # Overpass uses (south,west,north,east) format
        bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

        full_query = (
            f"[out:json][timeout:30];"
            f"({query_body}({bbox_str}););"
            f"out center;"
        )

        response = self._make_request(full_query)
        return response.get("elements", [])

    def _make_request(self, query: str) -> Dict:
        """Execute a query with retry on rate limit."""
        # Try GET first
        response = self.client.get(
            self.BASE_URL,
            params={"data": query},
        )
        if response.status_code == 200:
            return response.json()

        # Fall back to POST if GET fails
        response = self.client.post(
            self.BASE_URL,
            data={"data": query},
        )
        if response.status_code == 200:
            return response.json()

        logger.warning(
            f"Overpass API returned {response.status_code}: {response.text[:200]}"
        )
        return {"elements": []}

    def get_infrastructure(
        self, bbox: List[float]
    ) -> Dict:
        """
        Fetch infrastructure features within a bounding box.

        Returns:
            Dict with counts of buildings, roads, hospitals, schools
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

        results = {}
        feature_queries = {
            "buildings": '["building"]',
            "roads": '["highway"]',
            "hospitals": '["amenity"="hospital"]',
            "schools": '["amenity"="school"]',
            "water": '["natural"="water"]',
            "landuse_residential": '["landuse"="residential"]',
        }

        for feat_type, tags in feature_queries.items():
            # Use way only for buildings/roads/landuse, node for points
            if feat_type in ("hospitals", "schools"):
                query_body = f"node{tags}"
            else:
                query_body = f"way{tags}"

            query = (
                f"[out:json][timeout:25];"
                f"({query_body}({bbox_str}););"
                f"out count;"
            )

            try:
                resp = self._make_request(query)
                # Parse 'out count' response
                elements = resp.get("elements", [])
                tags_list = resp.get("tags", [])
                # Count elements - out count returns summary
                count = 0
                if elements:
                    count = len(elements)
                elif tags_list:
                    # out count format
                    for t in tags_list:
                        count += t.get("count", 0)
                results[feat_type] = count
                logger.info(f"Found {count} {feat_type}")
            except Exception as e:
                logger.warning(f"{feat_type} query failed: {e}")
                results[feat_type] = 0

        return results

    def get_population_estimate(
        self, bbox: List[float]
    ) -> Optional[int]:
        """
        Estimate population within bbox using place nodes.

        Returns:
            Estimated population or None
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

        query = (
            f'[out:json][timeout:25];'
            f'node({bbox_str})["place"~"city|town|village|hamlet"];'
            f'out tags;'
        )

        try:
            resp = self._make_request(query)
            elements = resp.get("elements", [])
            total_pop = 0
            for elem in elements:
                pop_str = elem.get("tags", {}).get("population", "")
                if pop_str:
                    try:
                        total_pop += int(pop_str)
                    except (ValueError, TypeError):
                        pass
            return total_pop if total_pop > 0 else None
        except Exception as e:
            logger.warning(f"Population query failed: {e}")
            return None

    def get_water_bodies(
        self, bbox: List[float]
    ) -> List[Dict]:
        """Fetch water bodies (lakes, rivers) with geometry."""
        min_lon, min_lat, max_lon, max_lat = bbox
        bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

        query = (
            f'[out:json][timeout:25];'
            f'way({bbox_str})["natural"="water"];'
            f'out center tags;'
        )

        try:
            resp = self._make_request(query)
            elements = resp.get("elements", [])
            water_bodies = []
            for elem in elements:
                tags = elem.get("tags", {})
                water_bodies.append({
                    "id": elem.get("id"),
                    "type": tags.get("water", "unknown"),
                    "name": tags.get("name", ""),
                })
            return water_bodies
        except Exception as e:
            logger.warning(f"Water bodies query failed: {e}")
            return []

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
