# Unit tests for Disaster Monitoring via Satellite Imagery

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Spectral Index Tests
# ============================================================================


class TestSpectralIndices:
    """Tests for spectral index computation from real satellite bands."""

    def test_ndvi_calculation(self):
        from src.processing.spectral import compute_ndvi

        nir = np.array([[0.8, 0.6], [0.3, 0.1]])
        red = np.array([[0.1, 0.2], [0.5, 0.4]])
        ndvi = compute_ndvi(nir, red)
        assert ndvi.shape == (2, 2)
        assert np.all(ndvi >= -1) and np.all(ndvi <= 1)
        assert ndvi[0, 0] > 0.5  # High NDVI when NIR >> RED

    def test_ndvi_zero_division(self):
        from src.processing.spectral import compute_ndvi

        nir = np.zeros((3, 3))
        red = np.zeros((3, 3))
        ndvi = compute_ndvi(nir, red)
        assert np.all(ndvi == 0)

    def test_ndwi_calculation(self):
        from src.processing.spectral import compute_ndwi

        green = np.array([[0.7, 0.5], [0.2, 0.1]])
        nir = np.array([[0.1, 0.3], [0.6, 0.5]])
        ndwi = compute_ndwi(green, nir)
        assert ndwi.shape == (2, 2)
        assert ndwi[0, 0] > 0  # Water: GREEN > NIR
        assert ndwi[1, 0] < 0  # Non-water: NIR > GREEN

    def test_nbr_calculation(self):
        from src.processing.spectral import compute_nbr

        nir = np.array([[0.7, 0.3], [0.5, 0.2]])
        swir2 = np.array([[0.2, 0.5], [0.4, 0.6]])
        nbr = compute_nbr(nir, swir2)
        assert nbr.shape == (2, 2)
        assert nbr[1, 1] < 0  # Burned: NIR < SWIR2 -> negative NBR

    def test_mndwi_calculation(self):
        from src.processing.spectral import compute_mndwi

        green = np.array([[0.6, 0.4]])
        swir1 = np.array([[0.2, 0.5]])
        mndwi = compute_mndwi(green, swir1)
        assert mndwi.shape == (1, 2)

    def test_slope_from_dem(self):
        from src.processing.spectral import compute_slope_from_dem

        dem = np.array([[100, 200], [300, 400]], dtype=np.float32)
        slope = compute_slope_from_dem(dem, resolution=30.0)
        assert slope.shape == (2, 2)
        assert np.all(slope >= 0) and np.all(slope <= 90)

    def test_curvature_from_dem(self):
        from src.processing.spectral import compute_curvature_from_dem

        dem = np.array([[100, 200, 300], [150, 250, 350]], dtype=np.float32)
        curvature = compute_curvature_from_dem(dem, resolution=30.0)
        assert curvature.shape == (2, 3)


# ============================================================================
# Raster I/O and Resampling Tests
# ============================================================================


class TestRasterProcessing:
    """Tests for raster I/O, resampling, and clipping functions."""

    def test_read_geotiff(self):
        """read_geotiff should read a GeoTIFF and return data + metadata."""
        import rasterio
        from rasterio.transform import Affine
        from src.processing.spectral import read_geotiff

        data = np.array([[100, 200], [300, 400]], dtype=np.float32)
        transform = Affine(30, 0, 0, 0, -30, 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tif"
            with rasterio.open(
                path, 'w', driver='GTiff', height=2, width=2,
                count=1, dtype=np.float32, crs='EPSG:4326', transform=transform,
            ) as dst:
                dst.write(data, 1)

            arr, meta = read_geotiff(path)
            assert arr.shape == (2, 2)
            assert meta["crs"] is not None
            assert "transform" in meta
            assert "bounds" in meta
            assert "resolution" in meta

    def test_raster_to_array(self):
        """raster_to_array should read any GDAL-supported format with full metadata."""
        import rasterio
        from rasterio.transform import Affine
        from src.processing.spectral import raster_to_array

        data = np.random.rand(10, 10).astype(np.float32) * 1000
        transform = Affine(30, 0, 0, 0, -30, 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dem.tif"
            with rasterio.open(
                path, 'w', driver='GTiff', height=10, width=10,
                count=1, dtype=np.float32, crs='EPSG:4326', transform=transform,
            ) as dst:
                dst.write(data, 1)

            arr, meta = raster_to_array(path)
            assert arr.shape == (1, 10, 10)  # bands, rows, cols
            assert meta["driver"] == "GTiff"
            assert meta["count"] == 1
            assert meta["dtype"] == "float32"

    def test_resample_raster_nearest(self):
        """resample_raster should change array shape using nearest neighbor."""
        import rasterio
        from rasterio.transform import Affine
        from src.processing.spectral import resample_raster

        data = np.random.rand(20, 20).astype(np.float32)
        src_transform = Affine(30, 0, 0, 0, -30, 0)
        dst_transform = Affine(15, 0, 0, 0, -15, 0)  # 2x finer resolution

        result = resample_raster(
            data=data,
            transform=src_transform,
            src_crs="EPSG:4326",
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_shape=(40, 40),
            method="nearest",
        )
        assert result.shape == (40, 40)

    def test_resample_raster_bilinear(self):
        """resample_raster should change array shape using bilinear interpolation."""
        from rasterio.transform import Affine
        from src.processing.spectral import resample_raster

        data = np.random.rand(10, 10).astype(np.float32)
        src_transform = Affine(30, 0, 0, 0, -30, 0)
        dst_transform = Affine(15, 0, 0, 0, -15, 0)

        result = resample_raster(
            data=data,
            transform=src_transform,
            src_crs="EPSG:4326",
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_shape=(20, 20),
            method="bilinear",
        )
        assert result.shape == (20, 20)

    def test_clip_raster_to_bbox(self):
        """clip_raster_to_bbox should extract a sub-region from a raster array."""
        from rasterio.transform import Affine
        from src.processing.spectral import clip_raster_to_bbox

        # Create a 100x100 raster at 30m resolution covering (0,0) to (3000,3000)
        data = np.random.rand(100, 100).astype(np.float32)
        transform = Affine(30, 0, 0, 0, -30, 10000)  # 30m pixels, north-up

        # Clip to bbox (100m, 9700m, 200m, 9800m) in CRS coordinates
        clipped, new_transform = clip_raster_to_bbox(
            data, transform, bbox=(100, 9700, 200, 9800), crs="EPSG:32610"
        )
        assert clipped.shape[0] > 0
        assert clipped.shape[1] > 0
        assert clipped.shape[0] < data.shape[0] or clipped.shape[1] < data.shape[1]
# ============================================================================


class TestHazardModels:
    """Tests for physical hazard models."""

    def test_flood_hazard(self):
        from src.risk.hazard import HazardModel

        model = HazardModel()
        ndwi = np.array([[0, 0.5], [0, 0]], dtype=np.float32)
        dem = np.array([[100, 50], [200, 150]], dtype=np.float32)
        hazard = model.flood_hazard(ndwi, dem)
        assert hazard.shape == (2, 2)
        assert hazard[0, 1] >= 0.5  # Water pixel from NDWI

    def test_fire_hazard(self):
        from src.risk.hazard import HazardModel

        model = HazardModel()
        nbr = np.array([[-0.2, 0.1], [0.5, 0.8]], dtype=np.float32)
        ndvi = np.array([[0.1, 0.1], [0.5, 0.6]], dtype=np.float32)
        hazard = model.fire_hazard(nbr, ndvi)
        assert hazard.shape == (2, 2)
        assert hazard[0, 0] >= 0.5  # Burned pixel from NBR

    def test_drought_hazard(self):
        from src.risk.hazard import HazardModel

        model = HazardModel()
        tws = np.array([[-15, -5], [5, 10]], dtype=np.float32)
        ndvi_anom = np.array([[-0.3, -0.1], [0.1, 0.2]], dtype=np.float32)
        hazard = model.drought_hazard(tws, ndvi_anom)
        assert hazard.shape == (2, 2)
        assert hazard[0, 0] > hazard[1, 1]  # More negative TWS = more drought

    def test_landslide_hazard(self):
        from src.risk.hazard import HazardModel

        model = HazardModel()
        slope = np.array([[30, 10], [45, 5]], dtype=np.float32)
        curvature = np.array([[0.05, 0.01], [0.1, 0.02]], dtype=np.float32)
        tws = np.array([[10, -5], [15, 0]], dtype=np.float32)
        hazard = model.landslide_hazard(slope, curvature, tws)
        assert hazard.shape == (2, 2)
        assert hazard[1, 0] > hazard[1, 1]  # Steep + high TWS = more risk

    def test_earthquake_hazard(self):
        from src.risk.hazard import HazardModel

        model = HazardModel()
        hazard = model.earthquake_hazard(
            magnitude=6.5, depth_km=10.0, distance_km=50.0, shape=(50, 50)
        )
        assert hazard.shape == (50, 50)
        assert np.all(hazard >= 0) and np.all(hazard <= 1)


# ============================================================================
# Composite Risk Tests
# ============================================================================


class TestCompositeRisk:
    """Tests for composite risk assessment."""

    def test_flood_risk(self):
        from src.risk.composite import CompositeRiskModel

        model = CompositeRiskModel()
        ndwi = np.zeros((50, 50), dtype=np.float32)
        ndwi[20:30, 20:30] = 0.5  # Water body
        dem = np.linspace(100, 500, 2500).reshape(50, 50).astype(np.float32)
        result = model.compute_flood_risk(ndwi, dem)
        assert result.hazard_type == "flood"
        assert result.risk_score.shape == (50, 50)
        assert 0 <= result.summary()["mean_risk"] <= 1

    def test_fire_risk(self):
        from src.risk.composite import CompositeRiskModel

        model = CompositeRiskModel()
        nbr = np.zeros((50, 50), dtype=np.float32)
        nbr[10:20, 10:20] = -0.2  # Burned area
        ndvi = np.random.rand(50, 50).astype(np.float32) * 0.4
        result = model.compute_fire_risk(nbr, ndvi)
        assert result.hazard_type == "wildfire"
        assert result.risk_score.shape == (50, 50)

    def test_drought_risk(self):
        from src.risk.composite import CompositeRiskModel

        model = CompositeRiskModel()
        tws = np.random.randn(50, 50).astype(np.float32) * 10
        ndvi_anom = np.random.randn(50, 50).astype(np.float32) * 0.2
        result = model.compute_drought_risk(tws, ndvi_anom)
        assert result.hazard_type == "drought"
        assert result.risk_score.shape == (50, 50)

    def test_landslide_risk(self):
        from src.risk.composite import CompositeRiskModel

        model = CompositeRiskModel()
        slope = np.random.rand(50, 50).astype(np.float32) * 45
        curvature = np.random.rand(50, 50).astype(np.float32) * 0.1
        tws = np.random.randn(50, 50).astype(np.float32) * 5
        result = model.compute_landslide_risk(slope, curvature, tws)
        assert result.hazard_type == "landslide"
        assert result.risk_score.shape == (50, 50)

    def test_earthquake_risk(self):
        from src.risk.composite import CompositeRiskModel

        model = CompositeRiskModel()
        result = model.compute_earthquake_risk(
            magnitude=6.5, depth_km=10.0, distance_km=50.0, shape=(50, 50)
        )
        assert result.hazard_type == "earthquake"
        assert result.risk_score.shape == (50, 50)

    def test_risk_result_summary(self):
        from src.risk.composite import CompositeRiskModel

        model = CompositeRiskModel()
        ndwi = np.zeros((20, 20), dtype=np.float32)
        ndwi[5:15, 5:15] = 0.5
        dem = np.linspace(100, 500, 400).reshape(20, 20).astype(np.float32)
        result = model.compute_flood_risk(ndwi, dem)
        summary = result.summary()
        assert "mean_risk" in summary
        assert "max_risk" in summary
        assert "coverage_pct" in summary
        assert "risk_distribution" in summary
        total = sum(summary["risk_distribution"].values())
        assert total == 400


# ============================================================================
# Alert Manager Tests
# ============================================================================


class TestAlertManager:
    """Tests for alert management."""

    def test_add_rule(self):
        from src.alert.manager import AlertManager, AlertRule

        mgr = AlertManager()
        rule = AlertRule(hazard_type="flood", threshold=0.7, severity="CRITICAL")
        mgr.add_rule(rule)
        assert len(mgr.rules) == 1

    def test_check_triggers(self):
        from src.alert.manager import AlertManager, AlertRule
        from src.risk.composite import CompositeRiskModel

        mgr = AlertManager()
        rule = AlertRule(hazard_type="flood", threshold=0.5, severity="HIGH")
        mgr.add_rule(rule)

        model = CompositeRiskModel()
        ndwi = np.array([[0.8, 0.6], [0.4, 0.7]], dtype=np.float32)
        dem = np.array([[100, 50], [200, 150]], dtype=np.float32)
        result = model.compute_flood_risk(ndwi, dem)

        triggered = mgr.check_triggers([result])
        assert len(triggered) == 1
        assert triggered[0]["hazard_type"] == "flood"

    def test_check_triggers_below_threshold(self):
        from src.alert.manager import AlertManager, AlertRule
        from src.risk.composite import CompositeRiskModel

        mgr = AlertManager()
        rule = AlertRule(hazard_type="flood", threshold=0.99, severity="EXTREME")
        mgr.add_rule(rule)

        model = CompositeRiskModel()
        ndwi = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
        dem = np.array([[400, 450], [300, 350]], dtype=np.float32)
        result = model.compute_flood_risk(ndwi, dem)

        triggered = mgr.check_triggers([result])
        assert len(triggered) == 0


# ============================================================================
# Visualization Tests
# ============================================================================


class TestVisualization:
    """Tests for visualization functions."""

    def test_plot_risk_map(self):
        import matplotlib

        matplotlib.use("Agg")
        from src.viz.maps import plot_risk_map
        from src.risk.composite import RiskResult

        risk = np.random.rand(20, 20).astype(np.float32)
        hazard = np.random.rand(20, 20).astype(np.float32)
        exposure = np.random.rand(20, 20).astype(np.float32)
        vulnerability = np.random.rand(20, 20).astype(np.float32)
        result = RiskResult(
            timestamp=datetime.now(),
            hazard_type="flood",
            risk_score=risk,
            hazard_component=hazard,
            exposure_component=exposure,
            vulnerability_component=vulnerability,
            risk_level=(risk * 5).astype(np.uint8),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_risk_map(result, "Test", Path(tmpdir) / "test.png")
            assert path.exists()

    def test_plot_earthquake_map(self):
        import matplotlib

        matplotlib.use("Agg")
        from src.viz.maps import plot_earthquake_map

        events = [
            {
                "lon": -105.0,
                "lat": 40.0,
                "magnitude": 5.5,
                "depth_km": 10.0,
                "place": "Test",
            },
            {
                "lon": -104.8,
                "lat": 40.1,
                "magnitude": 4.2,
                "depth_km": 15.0,
                "place": "Test2",
            },
        ]
        bbox = [-105.5, 39.5, -104.5, 40.5]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_earthquake_map(events, bbox, Path(tmpdir) / "eq.png")
            assert path.exists()


# ============================================================================
# Data Acquisition Tests (mocked HTTP)
# ============================================================================


class TestDataAcquisition:
    """Tests for data acquisition modules with mocked HTTP."""

    def test_nasa_power_fetch(self):
        from src.acquisition.nasa_power import NASAPOWERClient

        client = NASAPOWERClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "properties": {
                "parameter": {
                    "PRECTOTCORR": {"20240601": 5.2, "20240602": 3.1},
                    "T2M": {"20240601": 25.3, "20240602": 26.1},
                }
            }
        }
        mock_response.raise_for_status = MagicMock()
        client.client.get = MagicMock(return_value=mock_response)

        data = client.fetch_daily(-105.0, 40.0, "2024-06-01", "2024-06-30")
        assert "properties" in data
        assert "PRECTOTCORR" in data["properties"]["parameter"]
        client.close()

    def test_usgs_earthquake_query(self):
        from src.acquisition.usgs_earthquake import USGSEarthquakeClient

        client = USGSEarthquakeClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "features": [
                {
                    "id": "test1",
                    "properties": {
                        "time": "2024-06-15T00:00:00Z",
                        "mag": 5.5,
                        "place": "Test Location",
                        "url": "https://example.com",
                    },
                    "geometry": {"coordinates": [-105.0, 40.0, 10.0]},
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        client.client.get = MagicMock(return_value=mock_response)

        events = client.query("2024-06-01", "2024-06-30", min_magnitude=4.0)
        assert len(events) == 1
        assert events[0]["magnitude"] == 5.5
        assert events[0]["place"] == "Test Location"
        client.close()

    def test_gdacs_alerts(self):
        from src.acquisition.gdacs import GDACSClient

        client = GDACSClient()
        mock_response = MagicMock()
        mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Earthquake M6.5 - Test Region</title>
              <link>https://www.gdacs.org</link>
              <description>Test earthquake alert</description>
              <pubDate>Mon, 15 Jun 2024 00:00:00 GMT</pubDate>
              <guid>evt-001</guid>
            </item>
          </channel>
        </rss>"""
        mock_response.raise_for_status = MagicMock()
        client.client.get = MagicMock(return_value=mock_response)

        alerts = client.get_current_alerts(days=7)
        assert len(alerts) == 1
        assert alerts[0]["title"] == "Earthquake M6.5 - Test Region"
        client.close()

    def test_chirps_daily_url(self):
        from src.acquisition.chirps import CHIRPSClient

        client = CHIRPSClient()
        url = client.get_daily_url("2024-06-15")
        assert "chirps-v2.0.2024.06.15" in url
        assert url.endswith(".tif.gz")
        client.close()

    def test_nasa_power_context_manager(self):
        from src.acquisition.nasa_power import NASAPOWERClient

        with NASAPOWERClient() as client:
            assert client.client is not None


# ============================================================================
# NASA GIBS Integration Tests
# ============================================================================


class TestNASAGIBSClient:
    """Integration tests for NASA GIBS satellite imagery client."""

    def test_get_tile_returns_valid_image(self):
        """GIBS should return JPEG bytes for MODIS TrueColor tiles."""
        from src.acquisition.nasa_gibs import NASAGIBSClient

        with NASAGIBSClient() as gibs:
            tile = gibs.get_tile("true_color", "2024-06-15", 2, 0, 0)
            assert len(tile) > 0, "Tile should not be empty"
            assert tile[:2] == b"\xff\xd8", "Should be JPEG format"

    def test_get_tile_invalid_layer(self):
        """Invalid layer should return empty bytes without crashing."""
        from src.acquisition.nasa_gibs import NASAGIBSClient

        with NASAGIBSClient() as gibs:
            tile = gibs.get_tile("nonexistent_layer", "2024-06-15", 0, 0, 0)
            assert len(tile) == 0

    def test_download_tile_range_bbox(self):
        """GIBS should download tiles for a bounding box."""
        from src.acquisition.nasa_gibs import NASAGIBSClient

        bbox = (-105.5, 39.5, -104.5, 40.5)  # Colorado Front Range
        with NASAGIBSClient() as gibs:
            tiles = gibs.download_tile_range(
                layer="true_color",
                date="2024-06-15",
                bbox=bbox,
                z=6,
                output_dir=Path("/tmp/test_gibs_integration"),
            )
            assert len(tiles) >= 1, "Should download at least one tile"
            assert all(t.suffix == ".jpg" for t in tiles)


# ============================================================================
# OSM Overpass Integration Tests
# ============================================================================


class TestOSMOverpassClient:
    """Integration tests for OpenStreetMap Overpass API client."""

    def test_get_infrastructure(self):
        """Should return infrastructure feature counts."""
        from src.acquisition.osm_overpass import OSMOverpassClient

        with OSMOverpassClient(timeout=60) as osm:
            result = osm.get_infrastructure([-105.5, 39.5, -104.5, 40.5])
            assert "buildings" in result
            assert "roads" in result
            assert "hospitals" in result
            assert "schools" in result
            assert isinstance(result["buildings"], int)

    def test_get_water_bodies(self):
        """Should return list of water bodies."""
        from src.acquisition.osm_overpass import OSMOverpassClient

        with OSMOverpassClient(timeout=60) as osm:
            result = osm.get_water_bodies([-105.5, 39.5, -104.5, 40.5])
            assert isinstance(result, list)


# ============================================================================
# Visualization Tests (with geographic extent)
# ============================================================================


class TestVisualizationGeo:
    """Tests for geographic extent support in visualization."""

    def test_plot_risk_map_with_bbox(self):
        """Risk map should support geographic bounding boxes."""
        import matplotlib
        matplotlib.use("Agg")
        from src.viz.maps import plot_risk_map
        from src.risk.composite import RiskResult

        risk = np.random.rand(20, 20).astype(np.float32)
        hazard = np.random.rand(20, 20).astype(np.float32)
        exposure = np.random.rand(20, 20).astype(np.float32)
        vulnerability = np.random.rand(20, 20).astype(np.float32)
        result = RiskResult(
            timestamp=datetime.now(),
            hazard_type="flood",
            risk_score=risk,
            hazard_component=hazard,
            exposure_component=exposure,
            vulnerability_component=vulnerability,
            risk_level=(risk * 5).astype(np.uint8),
        )
        # Should not raise with bbox
        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_risk_map(
                result, "Test", Path(tmpdir) / "test.png",
                bbox=[-105.5, 39.5, -104.5, 40.5]
            )
            assert path.exists()


# ============================================================================
# Pipeline Import Tests
# ============================================================================


class TestPipeline:
    """Tests for pipeline structure and imports."""

    def test_pipeline_import(self):
        from src.pipeline import run_pipeline, main

        assert callable(run_pipeline)
        assert callable(main)

    def test_all_acquisition_imports(self):
        from src.acquisition.nasa_power import NASAPOWERClient
        from src.acquisition.usgs_earthquake import USGSEarthquakeClient
        from src.acquisition.gdacs import GDACSClient
        from src.acquisition.chirps import CHIRPSClient
        from src.acquisition.nasa_gibs import NASAGIBSClient
        from src.acquisition.osm_overpass import OSMOverpassClient

    def test_all_processing_imports(self):
        from src.processing.spectral import (
            compute_ndvi,
            compute_ndwi,
            compute_nbr,
            compute_mndwi,
            compute_evi,
            compute_savi,
            compute_bai,
            compute_ndbi,
            compute_slope_from_dem,
            compute_curvature_from_dem,
            compute_tpi_from_dem,
            read_geotiff,
            read_multiband_geotiff,
            resample_raster,
            raster_to_array,
            clip_raster_to_bbox,
        )

    def test_all_risk_imports(self):
        from src.risk.hazard import HazardModel
        from src.risk.composite import CompositeRiskModel, RiskResult
        from src.risk.exposure import ExposureModel
        from src.risk.vulnerability import VulnerabilityModel

    def test_all_viz_imports(self):
        from src.viz.maps import (
            plot_risk_map,
            plot_hazard_components,
            plot_multi_hazard_summary,
            plot_spectral_indices,
            plot_earthquake_map,
            plot_precipitation_map,
            plot_terrain_analysis,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
