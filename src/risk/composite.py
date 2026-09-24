#!/usr/bin/env python3
"""
Composite Risk Model
Combines hazard, exposure, and vulnerability into composite risk scores.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from src.risk.hazard import HazardModel
from src.risk.exposure import ExposureModel
from src.risk.vulnerability import VulnerabilityModel


@dataclass
class RiskResult:
    """Composite risk assessment result."""

    timestamp: datetime
    hazard_type: str
    risk_score: np.ndarray
    hazard_component: np.ndarray
    exposure_component: np.ndarray
    vulnerability_component: np.ndarray
    risk_level: np.ndarray  # 0-5 discrete levels

    def summary(self) -> dict:
        """Get summary statistics."""
        valid = self.risk_score[~np.isnan(self.risk_score)]
        if len(valid) == 0:
            return {"mean_risk": 0, "max_risk": 0, "coverage_pct": 0}
        return {
            "mean_risk": float(np.mean(valid)),
            "max_risk": float(np.max(valid)),
            "coverage_pct": float(len(valid) / self.risk_score.size * 100),
            "risk_distribution": {
                "none": int(np.sum(self.risk_level == 0)),
                "low": int(np.sum(self.risk_level == 1)),
                "moderate": int(np.sum(self.risk_level == 2)),
                "high": int(np.sum(self.risk_level == 3)),
                "critical": int(np.sum(self.risk_level == 4)),
                "extreme": int(np.sum(self.risk_level == 5)),
            },
        }


class CompositeRiskModel:
    """Multi-hazard composite risk assessment."""

    def __init__(self):
        self.hazard_model = HazardModel()
        self.exposure_model = ExposureModel()
        self.vulnerability_model = VulnerabilityModel()

    def compute_flood_risk(
        self,
        ndwi: np.ndarray,
        dem: np.ndarray,
        tws_anomaly: Optional[np.ndarray] = None,
        precipitation: Optional[np.ndarray] = None,
        population: Optional[np.ndarray] = None,
        infrastructure: Optional[np.ndarray] = None,
    ) -> RiskResult:
        """Compute composite flood risk."""
        hazard = self.hazard_model.flood_hazard(ndwi, dem, tws_anomaly, precipitation)
        exposure = self.exposure_model.compute_exposure(
            population, infrastructure, hazard.shape
        )
        vulnerability = self.vulnerability_model.flood_vulnerability(dem)

        risk = hazard * exposure * vulnerability
        risk = np.clip(risk, 0, 1)

        return RiskResult(
            timestamp=datetime.now(),
            hazard_type="flood",
            risk_score=risk,
            hazard_component=hazard,
            exposure_component=exposure,
            vulnerability_component=vulnerability,
            risk_level=self._discretize_risk(risk),
        )

    def compute_fire_risk(
        self,
        nbr: np.ndarray,
        ndvi: np.ndarray,
        temperature: Optional[np.ndarray] = None,
        wind_speed: Optional[np.ndarray] = None,
        population: Optional[np.ndarray] = None,
        infrastructure: Optional[np.ndarray] = None,
    ) -> RiskResult:
        """Compute composite wildfire risk."""
        hazard = self.hazard_model.fire_hazard(nbr, ndvi, temperature, wind_speed)
        exposure = self.exposure_model.compute_exposure(
            population, infrastructure, hazard.shape
        )
        vulnerability = self.vulnerability_model.fire_vulnerability(ndvi)

        risk = hazard * exposure * vulnerability
        risk = np.clip(risk, 0, 1)

        return RiskResult(
            timestamp=datetime.now(),
            hazard_type="wildfire",
            risk_score=risk,
            hazard_component=hazard,
            exposure_component=exposure,
            vulnerability_component=vulnerability,
            risk_level=self._discretize_risk(risk),
        )

    def compute_drought_risk(
        self,
        tws_anomaly: np.ndarray,
        ndvi_anomaly: np.ndarray,
        precipitation: Optional[np.ndarray] = None,
        soil_moisture: Optional[np.ndarray] = None,
        population: Optional[np.ndarray] = None,
        agriculture: Optional[np.ndarray] = None,
    ) -> RiskResult:
        """Compute composite drought risk."""
        hazard = self.hazard_model.drought_hazard(
            tws_anomaly, ndvi_anomaly, precipitation, soil_moisture
        )
        exposure = self.exposure_model.compute_exposure(
            population, agriculture, hazard.shape
        )
        vulnerability = self.vulnerability_model.drought_vulnerability(ndvi_anomaly)

        risk = hazard * exposure * vulnerability
        risk = np.clip(risk, 0, 1)

        return RiskResult(
            timestamp=datetime.now(),
            hazard_type="drought",
            risk_score=risk,
            hazard_component=hazard,
            exposure_component=exposure,
            vulnerability_component=vulnerability,
            risk_level=self._discretize_risk(risk),
        )

    def compute_landslide_risk(
        self,
        slope: np.ndarray,
        curvature: np.ndarray,
        tws_anomaly: np.ndarray,
        precipitation: Optional[np.ndarray] = None,
        population: Optional[np.ndarray] = None,
        infrastructure: Optional[np.ndarray] = None,
    ) -> RiskResult:
        """Compute composite landslide risk."""
        hazard = self.hazard_model.landslide_hazard(
            slope, curvature, tws_anomaly, precipitation
        )
        exposure = self.exposure_model.compute_exposure(
            population, infrastructure, hazard.shape
        )
        vulnerability = self.vulnerability_model.landslide_vulnerability(
            slope, tws_anomaly
        )

        risk = hazard * exposure * vulnerability
        risk = np.clip(risk, 0, 1)

        return RiskResult(
            timestamp=datetime.now(),
            hazard_type="landslide",
            risk_score=risk,
            hazard_component=hazard,
            exposure_component=exposure,
            vulnerability_component=vulnerability,
            risk_level=self._discretize_risk(risk),
        )

    def compute_earthquake_risk(
        self,
        magnitude: float,
        depth_km: float,
        distance_km: float,
        shape: tuple,
        building_density: Optional[np.ndarray] = None,
        population_density: Optional[np.ndarray] = None,
    ) -> RiskResult:
        """Compute composite earthquake risk."""
        hazard = self.hazard_model.earthquake_hazard(
            magnitude, depth_km, distance_km, shape
        )

        if building_density is not None and population_density is not None:
            vulnerability = self.vulnerability_model.earthquake_vulnerability(
                building_density, population_density
            )
        else:
            vulnerability = np.full(shape, 0.5, dtype=np.float32)

        exposure = self.exposure_model.compute_exposure(
            population_density, building_density, shape
        )

        risk = hazard * exposure * vulnerability
        risk = np.clip(risk, 0, 1)

        return RiskResult(
            timestamp=datetime.now(),
            hazard_type="earthquake",
            risk_score=risk,
            hazard_component=hazard,
            exposure_component=exposure,
            vulnerability_component=vulnerability,
            risk_level=self._discretize_risk(risk),
        )

    @staticmethod
    def _discretize_risk(risk: np.ndarray) -> np.ndarray:
        """Convert continuous risk to discrete levels (0-5)."""
        levels = np.zeros_like(risk, dtype=np.uint8)
        levels[risk > 0.1] = 1  # LOW
        levels[risk > 0.25] = 2  # MODERATE
        levels[risk > 0.5] = 3  # HIGH
        levels[risk > 0.75] = 4  # CRITICAL
        levels[risk > 0.9] = 5  # EXTREME
        return levels
