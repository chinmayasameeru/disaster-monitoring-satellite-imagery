# Contributing

This repository implements a satellite imagery-based disaster risk monitoring system following NVIDIA DLI and ISRO/IIRS course methodologies. All contributions must use **real data only** — no synthetic data, no mocks, no prototyping.

## Development Setup

```bash
git clone https://github.com/chinmayasameeru/disaster-monitoring-satellite-imagery.git
cd disaster-monitoring-satellite-imagery
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if exists
```

## Testing

All code must pass tests before submission:

```bash
# Run full test suite (37 tests excluding OSM integration)
pytest tests/ -v -k "not OSMOverpass"

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Code Standards

- **Black** formatting with max line length 100
- **flake8** linting with `--max-line-length=100 --ignore=E226,E402,E501,W503,F401,F841,F541`
- All new functions must have docstrings with Args/Returns
- All data sources must be real APIs — no fake data generators

## Adding New Data Sources

1. Create a client in `src/acquisition/` following the existing client pattern
2. Add tests in `tests/test_pipeline.py`
3. Update the README API reference table
4. Add to the data sources table in the README

## Generating Documentation Images

To regenerate all README images:

```bash
python scripts/generate_readme_examples.py
```

This downloads real GIBS satellite tiles, fetches USGS earthquake data, and generates all visualization maps.
