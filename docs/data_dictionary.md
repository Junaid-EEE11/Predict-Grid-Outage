# Data Dictionary

## 1. EAGLE-I Outage Records
| Column Name | Type | Description | Units / Range | Missingness Strategy |
| :--- | :--- | :--- | :--- | :--- |
| `fips_code` | string (5-char) | US County Federal Information Processing Standard code | e.g., '37183' | Required, drop if invalid |
| `county` | string | County name | e.g., 'Wake' | Derived from FIPS |
| `state` | string (2-char) | US State postal code | e.g., 'NC' | Derived from FIPS |
| `run_start_time` | datetime UTC | Timestamp of 15-minute raw snapshot | ISO-8601 UTC | Hourly aggregated (max) |
| `sum` / `customers_out` | integer | Number of electric customers without power | $\ge 0$ | Flagged; forward-filled if $\le 3$h, else marked missing |
| `total_customers` | integer | Modeled electric customer count in county | $> 0$ | Census / EIA customer count baseline |
| `outage_fraction` | float | Ratio of `customers_out` to `total_customers` | $[0.0, 1.0]$ | Clipped at 1.0; severe if $\ge 0.01$ |

## 2. Weather Variables (ISD / Hourly Product)
| Column Name | Type | Description | Units / Range | Source |
| :--- | :--- | :--- | :--- | :--- |
| `temperature_c` | float | Ambient 2m air temperature | Celsius ($[-40, 55]$) | NOAA ISD / Automated Surface Obs |
| `dew_point_c` | float | 2m dew point temperature | Celsius ($[-40, 40]$) | NOAA ISD |
| `relative_humidity` | float | Calculated relative humidity | Percentage ($[0, 100]$) | Derived from T and Td |
| `wind_speed_ms` | float | 10m sustained wind speed | m/s ($[0, 75]$) | NOAA ISD |
| `wind_gust_ms` | float | Peak wind gust in observation window | m/s ($[0, 100]$) | NOAA ISD |
| `surface_pressure_hpa` | float | Station surface atmospheric pressure | hPa ($[850, 1080]$) | NOAA ISD |
| `precipitation_1h_mm` | float | Liquid precipitation in preceding hour | mm ($\ge 0$) | NOAA ISD / Radar / Gauge |
| `precipitation_accum_6h_mm` | float | 6-hour rolling liquid precipitation | mm ($\ge 0$) | Rolling sum ($t-6$ to $t$) |
| `precipitation_accum_24h_mm`| float | 24-hour rolling liquid precipitation | mm ($\ge 0$) | Rolling sum ($t-24$ to $t$) |

## 3. NOAA Storm Event Metadata (Evaluation Only)
| Column Name | Type | Description | Values |
| :--- | :--- | :--- | :--- |
| `episode_id` | integer | Unique storm episode identifier | Positive integer |
| `event_id` | integer | Unique county-level event identifier | Positive integer |
| `event_type` | string | Standardized storm category | 'Thunderstorm Wind', 'High Wind', 'Tropical Storm', 'Ice Storm', etc. |
| `begin_time` | datetime UTC | Onset timestamp of storm event | ISO-8601 UTC |
| `end_time` | datetime UTC | Conclusion timestamp of storm event | ISO-8601 UTC |
