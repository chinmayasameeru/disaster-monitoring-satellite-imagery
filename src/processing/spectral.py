#!/usr/bin/env python3
"""
Spectral Index Computation
Computes vegetation, water, burn, and built-up indices from satellite bands.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import rasterio
import logging

logger = logging.getLogger(__name__)


def read_geotiff(filepath: Path) -> Tuple[np.ndarray, dict]:
    """Read a GeoTIFF file and return (data, metadata)."""
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        meta = {
            "crs": str(src.crs),
            "transform": list(src.transform),
            "bounds": [
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            ],
            "resolution": src.res,
            "shape": data.shape,
            "nodata": src.nodata,
            "dtype": str(src.dtypes[0]),
        }
    return data, meta


def read_multiband_geotiff(filepath: Path) -> Tuple[np.ndarray, dict]:
    """Read a multi-band GeoTIFF and return (data, metadata) with shape (bands, rows, cols)."""
    with rasterio.open(filepath) as src:
        data = src.read().astype(np.float32)
        meta = {
            "crs": str(src.crs),
            "transform": list(src.transform),
            "bounds": [
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            ],
            "resolution": src.res,
            "shape": data.shape,
            "nodata": src.nodata,
            "band_count": src.count,
            "band_descriptions": src.descriptions,
        }
    return data, meta


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Vegetation Index.
    NDVI = (NIR - RED) / (NIR + RED)
    Range: [-1, 1], healthy vegetation > 0.3
    """
    nir = nir.astype(np.float64)
    red = red.astype(np.float64)
    denom = nir + red
    ndvi = np.where(denom != 0, (nir - red) / denom, 0)
    return np.clip(ndvi, -1, 1).astype(np.float32)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Water Index.
    NDWI = (GREEN - NIR) / (GREEN + NIR)
    Range: [-1, 1], water bodies > 0.3
    """
    green = green.astype(np.float64)
    nir = nir.astype(np.float64)
    denom = green + nir
    ndwi = np.where(denom != 0, (green - nir) / denom, 0)
    return np.clip(ndwi, -1, 1).astype(np.float32)


def compute_mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Compute Modified Normalized Difference Water Index.
    MNDWI = (GREEN - SWIR1) / (GREEN + SWIR1)
    Better for urban water body detection than NDWI.
    """
    green = green.astype(np.float64)
    swir1 = swir1.astype(np.float64)
    denom = green + swir1
    mndwi = np.where(denom != 0, (green - swir1) / denom, 0)
    return np.clip(mndwi, -1, 1).astype(np.float32)


def compute_nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Burn Ratio.
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    Range: [-1, 1], burned areas < 0.1
    """
    nir = nir.astype(np.float64)
    swir2 = swir2.astype(np.float64)
    denom = nir + swir2
    nbr = np.where(denom != 0, (nir - swir2) / denom, 0)
    return np.clip(nbr, -1, 1).astype(np.float32)


def compute_bai(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute Burn Area Index.
    BAI = 1 / ((0.1 - RED)^2 + (0.06 - NIR)^2)
    Higher values indicate more severely burned areas.
    """
    red = red.astype(np.float64)
    nir = nir.astype(np.float64)
    denom = (0.1 - red) ** 2 + (0.06 - nir) ** 2
    bai = np.where(denom != 0, 1.0 / denom, 0)
    return np.clip(bai, 0, 1000).astype(np.float32)


def compute_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Compute Enhanced Vegetation Index.
    EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
    Range: [-1, 1], better than NDVI in high biomass areas.
    """
    nir = nir.astype(np.float64)
    red = red.astype(np.float64)
    blue = blue.astype(np.float64)
    denom = nir + 6 * red - 7.5 * blue + 1
    evi = np.where(denom != 0, 2.5 * (nir - red) / denom, 0)
    return np.clip(evi, -1, 1).astype(np.float32)


def compute_savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """
    Compute Soil Adjusted Vegetation Index.
    SAVI = (1 + L) * (NIR - RED) / (NIR + RED + L)
    L = 0.5 for intermediate vegetation cover.
    """
    nir = nir.astype(np.float64)
    red = red.astype(np.float64)
    denom = nir + red + L
    savi = np.where(denom != 0, (1 + L) * (nir - red) / denom, 0)
    return np.clip(savi, -1, 1).astype(np.float32)


def compute_ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Built-up Index.
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    Range: [-1, 1], built-up areas > 0
    """
    swir1 = swir1.astype(np.float64)
    nir = nir.astype(np.float64)
    denom = swir1 + nir
    ndbi = np.where(denom != 0, (swir1 - nir) / denom, 0)
    return np.clip(ndbi, -1, 1).astype(np.float32)


def compute_slope_from_dem(dem: np.ndarray, resolution: float = 30.0) -> np.ndarray:
    """
    Compute slope in degrees from a Digital Elevation Model.

    Args:
        dem: 2D elevation array (meters)
        resolution: Pixel resolution in meters (default 30m for SRTM)

    Returns:
        Slope in degrees [0, 90]
    """
    dy, dx = np.gradient(dem, resolution)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    return slope.astype(np.float32)


def compute_curvature_from_dem(dem: np.ndarray, resolution: float = 30.0) -> np.ndarray:
    """
    Compute curvature from a Digital Elevation Model.

    Args:
        dem: 2D elevation array (meters)
        resolution: Pixel resolution in meters

    Returns:
        Curvature array (positive = convex, negative = concave)
    """
    dy, dx = np.gradient(dem, resolution)
    d2y, _ = np.gradient(dy, resolution)
    _, d2x = np.gradient(dx, resolution)
    curvature = d2x + d2y
    return curvature.astype(np.float32)


def compute_tpi_from_dem(dem: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Compute Topographic Position Index (TPI).
    TPI = elevation - mean elevation of surrounding cells.

    Args:
        dem: 2D elevation array
        window_size: Size of the moving window (default 3x3)

    Returns:
        TPI array (positive = ridge, negative = valley)
    """
    from scipy.ndimage import uniform_filter

    mean_elev = uniform_filter(dem.astype(np.float64), size=window_size)
    tpi = dem.astype(np.float64) - mean_elev
    return tpi.astype(np.float32)


def resample_raster(
    data: np.ndarray,
    transform,
    src_crs: str,
    dst_transform,
    dst_crs: str,
    dst_shape: tuple,
    method: str = "bilinear",
) -> np.ndarray:
    """
    Resample a raster array to a new resolution and CRS.

    Uses GDAL's reproject function with configurable resampling methods.

    Args:
        data: Input 2D raster array
        transform: Source affine transform (rasterio.transform.Affine)
        src_crs: Source coordinate reference system (WKT or EPSG string)
        dst_transform: Destination affine transform
        dst_crs: Destination CRS
        dst_shape: Target (rows, cols)
        method: Resampling method ('nearest', 'bilinear', 'cubic', 'lanczos')

    Returns:
        Resampled 2D array with dst_shape
    """
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    method_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "lanczos": Resampling.lanczos,
    }
    resampling = method_map.get(method, Resampling.bilinear)

    dst = np.empty(dst_shape, dtype=data.dtype)
    reproject(
        source=data,
        destination=dst,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    return dst


def raster_to_array(filepath: Path) -> Tuple[np.ndarray, dict]:
    """
    Read any raster format supported by GDAL (GeoTIFF, JPEG2000, NetCDF, etc.)
    and return the data array with full geotransform and CRS metadata.

    Uses rasterio (GDAL bindings) for reading diverse geospatial raster formats.

    Args:
        filepath: Path to raster file

    Returns:
        Tuple of (data array, metadata dict with crs, transform, bounds, res)
    """
    with rasterio.open(filepath) as src:
        data = src.read().astype(np.float32)
        meta = {
            "crs": str(src.crs),
            "transform": list(src.transform),
            "bounds": [
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            ],
            "resolution": src.res,
            "shape": src.shape,
            "nodata": src.nodata,
            "dtype": str(src.dtypes[0]),
            "count": src.count,
            "driver": src.driver,
        }
    return data, meta


def clip_raster_to_bbox(
    data: np.ndarray,
    transform,
    bbox: Tuple[float, float, float, float],
    crs: str,
) -> Tuple[np.ndarray, tuple]:
    """
    Clip a raster array to a bounding box using GDAL's window calculation.

    Args:
        data: 2D raster array
        transform: Rasterio affine transform (rasterio.transform.Affine)
        bbox: (min_lon, min_lat, max_lon, max_lat)
        crs: CRS of the raster

    Returns:
        (clipped_data, clipped_transform)
    """
    from rasterio.transform import rowcol
    from affine import Affine

    min_lon, min_lat, max_lon, max_lat = bbox
    row_min, col_min = rowcol(transform, min_lon, min_lat)
    row_max, col_max = rowcol(transform, max_lon, max_lat)

    # Ensure proper ordering (row/col may come back swapped depending on transform)
    row_start = min(row_min, row_max)
    row_end = max(row_min, row_max)
    col_start = min(col_min, col_max)
    col_end = max(col_min, col_max)

    # Clamp to array bounds
    row_start = max(0, min(row_start, data.shape[0] - 1))
    row_end = max(0, min(row_end, data.shape[0] - 1))
    col_start = max(0, min(col_start, data.shape[1] - 1))
    col_end = max(0, min(col_end, data.shape[1] - 1))

    clipped = data[row_start:row_end + 1, col_start:col_end + 1]
    # Compute new transform: shift origin to the top-left of the clipped window
    new_transform = transform @ Affine.translation(col_start, row_start)
    return clipped, new_transform
