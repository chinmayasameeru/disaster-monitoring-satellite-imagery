#!/usr/bin/env python3
"""
Visualization Module
Creates publication-quality maps from real risk data.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from typing import Optional, List
from src.risk.composite import RiskResult

# Risk level colors (NASA-style)
RISK_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#9b59b6", "#2c3e50"]
RISK_LABELS = ["None", "Low", "Moderate", "High", "Critical", "Extreme"]

# Hazard-specific colormaps
FLOOD_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "flood", ["#ffffff", "#3498db", "#2c3e50"]
)
FIRE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "fire", ["#2ecc71", "#f1c40f", "#e74c3c", "#2c3e50"]
)
DROUGHT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "drought", ["#27ae60", "#f39c12", "#d35400", "#7b241c"]
)
LANDSLIDE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "landslide", ["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"]
)
EARTHQUAKE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "earthquake", ["#2ecc71", "#f1c40f", "#e74c3c", "#8b0000"]
)


def plot_risk_map(
    result: RiskResult, title: str, output_path: Path,
    dem: Optional[np.ndarray] = None,
    bbox: Optional[List[float]] = None,
) -> Path:
    """Plot a risk map with colorbar and legend.

    Args:
        result: RiskResult object
        title: Map title
        output_path: Save path
        dem: Optional DEM for hillshade overlay
        bbox: [min_lon, min_lat, max_lon, max_lat] for geographic axes
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Choose colormap based on hazard type
    cmap_map = {
        "flood": FLOOD_CMAP,
        "wildfire": FIRE_CMAP,
        "drought": DROUGHT_CMAP,
        "landslide": LANDSLIDE_CMAP,
        "earthquake": EARTHQUAKE_CMAP,
    }
    cmap = cmap_map.get(result.hazard_type, plt.cm.YlOrRd)

    # Set geographic extent if bbox provided
    extent = None
    if bbox is not None:
        extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
    else:
        ax.set_xlabel("Longitude (pixels)", fontsize=12)
        ax.set_ylabel("Latitude (pixels)", fontsize=12)

    # Plot risk score with geographic extent
    im = ax.imshow(
        result.risk_score, cmap=cmap, vmin=0, vmax=1, extent=extent
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Risk Score", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Add risk level legend
    legend_elements = [
        Patch(facecolor=RISK_COLORS[i], label=RISK_LABELS[i]) for i in range(6)
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.9)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude (pixels)", fontsize=12)
    ax.set_ylabel("Latitude (pixels)", fontsize=12)

    # Add summary text
    summary = result.summary()
    summary_text = (
        f"Mean Risk: {summary['mean_risk']:.3f}\n"
        f"Max Risk: {summary['max_risk']:.3f}\n"
        f"Coverage: {summary['coverage_pct']:.1f}%"
    )
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_hazard_components(result: RiskResult, title: str, output_path: Path) -> Path:
    """Plot hazard, exposure, vulnerability, and composite risk side by side."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    im0 = axes[0].imshow(result.hazard_component, cmap="YlOrRd", vmin=0, vmax=1)
    axes[0].set_title("Hazard", fontsize=14, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(result.exposure_component, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Exposure", fontsize=14, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(result.vulnerability_component, cmap="Oranges", vmin=0, vmax=1)
    axes[2].set_title("Vulnerability", fontsize=14, fontweight="bold")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(result.risk_score, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[3].set_title("Composite Risk", fontsize=14, fontweight="bold")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_multi_hazard_summary(results: List[RiskResult], output_path: Path) -> Path:
    """Plot multi-hazard summary dashboard."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 8))

    if n == 1:
        axes = [axes]

    for i, result in enumerate(results):
        im = axes[i].imshow(result.risk_score, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[i].set_title(
            result.hazard_type.replace("_", " ").title(), fontsize=14, fontweight="bold"
        )
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        summary = result.summary()
        text = f"Mean: {summary['mean_risk']:.3f}\nMax: {summary['max_risk']:.3f}"
        axes[i].text(
            0.02,
            0.98,
            text,
            transform=axes[i].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.suptitle("Multi-Hazard Risk Assessment", fontsize=18, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_spectral_indices(
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    nbr: np.ndarray,
    dem: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot spectral indices comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    im0 = axes[0, 0].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[0, 0].set_title("NDVI (Vegetation Health)", fontsize=14, fontweight="bold")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(ndwi, cmap="BrBG", vmin=-1, vmax=1)
    axes[0, 1].set_title("NDWI (Water Bodies)", fontsize=14, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(nbr, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1, 0].set_title("NBR (Burn Severity)", fontsize=14, fontweight="bold")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(dem, cmap="terrain")
    axes[1, 1].set_title("Digital Elevation Model", fontsize=14, fontweight="bold")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle("Spectral Indices and Terrain", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_earthquake_map(
    events: list,
    bbox: List[float],
    output_path: Path,
    title: str = "USGS Earthquake Events",
) -> Path:
    """
    Plot earthquake events on a map.

    Args:
        events: List of earthquake dicts from USGSEarthquakeClient
        bbox: [min_lon, min_lat, max_lon, max_lat]
        output_path: Output path
        title: Map title
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    lons = [e["lon"] for e in events if e.get("lon") is not None]
    lats = [e["lat"] for e in events if e.get("lat") is not None]
    mags = [e["magnitude"] for e in events if e.get("magnitude") is not None]
    depths = [e["depth_km"] for e in events if e.get("depth_km") is not None]

    if lons and lats:
        # Color by depth, size by magnitude
        sizes = [max(20, m**2 * 10) for m in mags]
        if depths:
            colors = depths
            cmap = plt.cm.viridis_r  # Reverse so shallow = more visible
        else:
            colors = "red"
            cmap = None

        scatter = ax.scatter(
            lons,
            lats,
            c=colors,
            s=sizes,
            cmap=cmap,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )
        if depths:
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Depth (km)", fontsize=12)

        # Annotate significant events
        for event in events:
            if event.get("magnitude") and event["magnitude"] >= 6.0:
                ax.annotate(
                    f"M{event['magnitude']:.1f}\n{event.get('place', '')[:30]}",
                    xy=(event["lon"], event["lat"]),
                    fontsize=8,
                    fontweight="bold",
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add magnitude legend
    if mags:
        max_mag = max(mags) if mags else 0
        ax.text(
            0.02,
            0.02,
            f"Events: {len(events)}\nMax Magnitude: {max_mag:.1f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_terrain_analysis(
    dem: np.ndarray,
    slope: np.ndarray,
    tpi: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot terrain analysis: elevation, slope, TPI, and hillshade.

    Args:
        dem: Digital Elevation Model array (meters)
        slope: Slope array (degrees)
        tpi: Topographic Position Index array
        output_path: Save path

    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Compute hillshade from DEM for context
    angle = np.pi / 4  # 45 degree illumination
    dx, dy = np.gradient(dem.astype(np.float64))
    hillshade = np.cos(angle) * dy + np.sin(angle) * dx
    hillshade = np.clip(hillshade, 0, None)
    # Normalize hillshade to [0, 1]
    if hillshade.max() > 0:
        hillshade = hillshade / hillshade.max()

    im0 = axes[0, 0].imshow(dem, cmap="terrain")
    axes[0, 0].set_title("Digital Elevation Model (m)", fontsize=14, fontweight="bold")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(slope, cmap="YlOrRd")
    axes[0, 1].set_title("Slope (degrees)", fontsize=14, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(tpi, cmap="RdBu_r", vmin=-np.percentile(np.abs(tpi), 95),
                            vmax=np.percentile(np.abs(tpi), 95))
    axes[1, 0].set_title("Topographic Position Index", fontsize=14, fontweight="bold")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(hillshade, cmap="gray")
    axes[1, 1].set_title("Hillshade", fontsize=14, fontweight="bold")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle("Terrain Analysis — Big Sur Coastal Range", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_precipitation_map(
    precip_data: np.ndarray,
    bbox: List[float],
    output_path: Path,
    title: str = "CHIRPS Precipitation",
    units: str = "mm/day",
) -> Path:
    """Plot precipitation data on a map."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
    im = ax.imshow(
        precip_data, cmap="YlGnBu", vmin=0, extent=extent
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(units, fontsize=12)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude (pixels)", fontsize=12)
    ax.set_ylabel("Latitude (pixels)", fontsize=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
