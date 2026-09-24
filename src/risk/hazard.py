#!/usr/bin/env python3
"""
Physical Hazard Models
Computes hazard scores from real satellite-derived data.
"""

import numpy as np
from scipy.ndimage import zoom
from typing import Optional


class HazardModel:
    """Physical hazard models for flood, fire, drought, and landslide."""

    def flood_hazard(
        self,
        ndwi: np.ndarray,
        dem: np.ndarray,
        tws_anomaly: Optional[np.ndarray] = None,
        precipitation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Flood hazard from NDWI (water extent), DEM (low-lying areas),
        TWS anomaly (groundwater saturation), and precipitation.

        Returns:
            Hazard score [0, 1]
        """
        # Water extent from NDWI
        water_mask = (ndwi > 0.3).astype(np.float32)

        # Low elevation = higher flood risk
        elev_min, elev_max = np.min(dem), np.max(dem)
        if elev_max > elev_min:
            elev_risk = 1.0 - (dem - elev_min) / (elev_max - elev_min)
        else:
            elev_risk = np.zeros_like(dem, dtype=np.float32)

        hazard = np.maximum(water_mask, elev_risk * 0.4)

        # TWS anomaly (positive = saturated ground)
        if tws_anomaly is not None:
            tws_resized = self._resize_to_match(tws_anomaly, hazard.shape)
            tws_norm = np.clip((tws_resized + 10) / 30, 0, 1)
            hazard = np.maximum(hazard, tws_norm * 0.5)

        # Heavy precipitation
        if precipitation is not None:
            precip_resized = self._resize_to_match(precipitation, hazard.shape)
            # Normalize: > 50mm/day is extreme
            precip_norm = np.clip(precip_resized / 50.0, 0, 1)
            hazard = np.maximum(hazard, precip_norm * 0.6)

        return np.clip(hazard, 0, 1).astype(np.float32)

    def fire_hazard(
        self,
        nbr: np.ndarray,
        ndvi: np.ndarray,
        temperature: Optional[np.ndarray] = None,
        wind_speed: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fire hazard from NBR (burn severity), NDVI (fuel dryness),
        temperature, and wind speed.

        Returns:
            Hazard score [0, 1]
        """
        # Burned area from NBR
        burned_mask = (nbr < 0.1).astype(np.float32)

        # Dry fuel from low NDVI
        dry_fuel = np.clip(1 - (ndvi + 1) / 2, 0, 1)

        hazard = np.maximum(burned_mask, dry_fuel * 0.3)

        # High temperature
        if temperature is not None:
            temp_resized = self._resize_to_match(temperature, hazard.shape)
            # > 35°C is high fire risk
            temp_norm = np.clip((temp_resized - 25) / 20, 0, 1)
            hazard = np.maximum(hazard, temp_norm * 0.5)

        # High wind speed
        if wind_speed is not None:
            wind_resized = self._resize_to_match(wind_speed, hazard.shape)
            # > 10 m/s is high fire risk
            wind_norm = np.clip(wind_resized / 15.0, 0, 1)
            hazard = np.maximum(hazard, wind_norm * 0.4)

        return np.clip(hazard, 0, 1).astype(np.float32)

    def drought_hazard(
        self,
        tws_anomaly: np.ndarray,
        ndvi_anomaly: np.ndarray,
        precipitation: Optional[np.ndarray] = None,
        soil_moisture: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Drought hazard from TWS anomaly, NDVI anomaly,
        precipitation deficit, and soil moisture.

        Returns:
            Hazard score [0, 1]
        """
        # Negative TWS = water deficit
        tws_drought = np.clip(-tws_anomaly / 20, 0, 1)

        # Negative NDVI anomaly = vegetation stress
        ndvi_drought = np.clip(-ndvi_anomaly * 3, 0, 1)

        hazard = np.maximum(tws_drought, ndvi_drought)

        # Precipitation deficit
        if precipitation is not None:
            precip_resized = self._resize_to_match(precipitation, hazard.shape)
            # < 1mm/day is drought conditions
            precip_drought = np.clip(1 - precip_resized / 5.0, 0, 1)
            hazard = np.maximum(hazard, precip_drought * 0.5)

        # Low soil moisture
        if soil_moisture is not None:
            sm_resized = self._resize_to_match(soil_moisture, hazard.shape)
            # < 0.1 m³/m³ is very dry
            sm_drought = np.clip(1 - sm_resized / 0.2, 0, 1)
            hazard = np.maximum(hazard, sm_drought * 0.6)

        return np.clip(hazard, 0, 1).astype(np.float32)

    def landslide_hazard(
        self,
        slope: np.ndarray,
        curvature: np.ndarray,
        tws_anomaly: np.ndarray,
        precipitation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Landslide hazard from slope, curvature, TWS anomaly (water loading),
        and precipitation.

        Returns:
            Hazard score [0, 1]
        """
        # Slope (steeper = more risk)
        slope_norm = np.clip(slope / 45.0, 0, 1)

        # Curvature (convex = more risk)
        curv_abs = np.abs(curvature)
        curv_p95 = np.percentile(curv_abs, 95)
        if curv_p95 > 0:
            curv_norm = np.clip(curv_abs / curv_p95, 0, 1)
        else:
            curv_norm = np.zeros_like(curv_abs, dtype=np.float32)

        # TWS anomaly (positive = water loading)
        tws_resized = self._resize_to_match(tws_anomaly, slope.shape)
        tws_norm = np.clip((tws_resized + 5) / 25, 0, 1)

        # Weighted combination
        hazard = 0.45 * slope_norm + 0.25 * curv_norm + 0.30 * tws_norm

        # Heavy precipitation trigger
        if precipitation is not None:
            precip_resized = self._resize_to_match(precipitation, slope.shape)
            # > 30mm/day is a landslide trigger
            precip_norm = np.clip(precip_resized / 50.0, 0, 1)
            hazard = np.maximum(hazard, precip_norm * 0.7)

        return np.clip(hazard, 0, 1).astype(np.float32)

    def earthquake_hazard(
        self,
        magnitude: float,
        depth_km: float,
        distance_km: float,
        shape: tuple,
    ) -> np.ndarray:
        """
        Earthquake hazard from magnitude, depth, and distance.
        Uses a simplified attenuation model.

        Args:
            magnitude: Earthquake magnitude (Mw)
            depth_km: Hypocenter depth in km
            distance_km: Distance from epicenter in km
            shape: Output array shape (rows, cols)

        Returns:
            Hazard score [0, 1]
        """
        # Simplified MMI attenuation
        # MMI = c1 + c2 * M - c3 * log10(R + c4) - c5 * R
        c1, c2, c3, c4, c5 = 1.5, 1.3, 1.0, 10.0, 0.0015
        r = np.sqrt(distance_km**2 + depth_km**2)
        mmi = c1 + c2 * magnitude - c3 * np.log10(r + c4) - c5 * r
        mmi = np.clip(mmi, 1, 12)

        # Normalize MMI to [0, 1]
        hazard_value = np.clip((mmi - 3) / 9, 0, 1)
        hazard = np.full(shape, hazard_value, dtype=np.float32)

        return hazard

    @staticmethod
    def _resize_to_match(data: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize array to match target shape using bilinear interpolation."""
        if data.shape == target_shape:
            return data
        factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
        return zoom(data, factors, order=1)
