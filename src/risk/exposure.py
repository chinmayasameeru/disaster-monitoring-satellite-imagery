#!/usr/bin/env python3
"""
Exposure Models
Population, infrastructure, and economic exposure assessment.
"""

import numpy as np
from typing import Optional


class ExposureModel:
    """Exposure assessment using real-world data."""

    def compute_exposure(
        self,
        population: Optional[np.ndarray] = None,
        infrastructure: Optional[np.ndarray] = None,
        target_shape: Optional[tuple] = None,
    ) -> np.ndarray:
        """
        Compute composite exposure score [0, 1].

        Args:
            population: Population density array (people/km²)
            infrastructure: Infrastructure density index
            target_shape: Output shape if no input data

        Returns:
            Exposure score array [0, 1]
        """
        if population is not None:
            # Log-scale normalization for population
            pop_max = np.max(population)
            if pop_max > 0:
                pop_norm = np.log1p(population) / np.log1p(pop_max)
                pop_norm = np.clip(pop_norm, 0, 1)
            else:
                pop_norm = np.zeros_like(population, dtype=np.float32)
        else:
            pop_norm = None

        if infrastructure is not None:
            # Normalize infrastructure index
            infra_p95 = np.percentile(infrastructure, 95)
            if infra_p95 > 0:
                infra_norm = np.clip(infrastructure / infra_p95, 0, 1)
            else:
                infra_norm = np.zeros_like(infrastructure, dtype=np.float32)
        else:
            infra_norm = None

        if pop_norm is not None and infra_norm is not None:
            return np.maximum(pop_norm, infra_norm)
        elif pop_norm is not None:
            return pop_norm
        elif infra_norm is not None:
            return infra_norm
        else:
            if target_shape is None:
                target_shape = (100, 100)
            return np.full(target_shape, 0.5, dtype=np.float32)

    def compute_population_exposure(
        self,
        population: np.ndarray,
        hazard_mask: np.ndarray,
    ) -> float:
        """
        Compute population exposure: sum of population in hazard-affected areas.

        Returns:
            Total exposed population
        """
        exposed = population * hazard_mask
        return float(np.sum(exposed))
