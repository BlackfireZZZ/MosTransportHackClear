# Moscow tram weather, 2025

Source: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api),
[CC BY 4.0](https://open-meteo.com/en/licence). We shifted preceding-hour
precipitation, rain and snowfall from API timestamp H+1 to boarding hour H,
deduplicated the returned grid cells and aggregated them by route. The
[manifest](manifest.json) records source coordinates, units, coverage and
SHA-256 checksums.
The 2025 collection used 45 HTTP requests, estimated at 688 weighted calls;
the exporter paces fresh requests below 480 weighted calls per minute.

`stops.csv` maps 445 OSM `stop_position` nodes on the ten scored routes to the
weather grid cell returned when each node's coordinate was queried. The nodes
have 225 distinct stop names. `zones.csv` records both the requested
representative coordinate and the actual API grid coordinate of each of the
nine cells. `zone-weather.csv.gz` contains 78,840 cell-hours.

`route-zones.csv` counts a route's OSM nodes in each cell. For each of 87,600
route-hours, `route-weather.csv.gz` contains the stop-count-weighted mean of
continuous weather fields and weighted mode of `weather_code`. All ten fields
are present for every route-hour. A stop's hourly weather is obtained by
joining `stops.zone_id` to `zone-weather.zone_id` at `(date,hour)`.

The stop source is the repository's 2026-09-18 OSM snapshot. It provides a
spatial proxy for 2025 routes, not the observed boarding stop of any validation.
Actual November–December 2025 weather is a retrospective feature and was not
known at the 2025-10-31 forecast origin. The API grid coordinate can be up to
8.1 km from an OSM node in this export. The route aggregation gives equal
weight to stop nodes, not observed passenger exposure.
