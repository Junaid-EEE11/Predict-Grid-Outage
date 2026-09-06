# Data Provenance

## Upstream Data Sources

1. **EAGLE-I (Environment for the Analysis of Geo-Located Energy Information)**
   - **Provider**: U.S. Department of Energy (DOE) & Oak Ridge National Laboratory (ORNL).
   - **Nature**: 15-minute national distribution outage monitoring system compiled from utility public status pages.
   - **Processing**: Resampled to hourly intervals using the maximum customer count per hour.
   - **Missingness Note**: Gaps in reporting are not assumed to be zero outages; they are explicitly flagged with `coverage_flag` and `is_imputed`.

2. **NOAA Integrated Surface Database (ISD) & Hourly Observations**
   - **Provider**: National Centers for Environmental Information (NCEI), National Oceanic and Atmospheric Administration (NOAA).
   - **Nature**: Hourly meteorological observations from global surface stations.
   - **Geospatial Mapping**: Station observations mapped to county centroids using nearest-station and inverse distance weighting (IDW) interpolation.

3. **NOAA Storm Events Database**
   - **Provider**: NOAA / NCEI.
   - **Nature**: Curated meteorological event records documenting storm types, durations, and impacts.
   - **Usage Rule**: Strictly used as post-event evaluation metadata and episode stratification. Never used as predictive input features.

4. **U.S. Census Bureau County Cartographic Boundaries**
   - **Provider**: U.S. Census Bureau TIGER/Line Geodatabase.
   - **Nature**: Official county boundary shapefiles used for queen contiguity graph generation and spatial adjacency matrix construction.
