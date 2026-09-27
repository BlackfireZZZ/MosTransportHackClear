# Moscow issued weather, January–October 2025

Source: [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api),
Moscow city-centre coordinate 55.7558, 37.6173. The API stitches early hours
of previously issued operational forecast runs. This is a citywide proxy, not a
route-stop measurement or a 61-day forecast. Two fields, temperature and
precipitation, cover all 7,296 Moscow civil hours from January 1 to October 31.
The exact request URL, source-response SHA-256, compact gzip SHA-256, date range,
timezone and transformation are in [manifest.json](manifest.json).

The model uses only values for dates at least two days before each forecast
origin, allowing for forecast run processing time. November–December 2025
actual weather is absent. No later historical-weather reanalysis enters the
approved source-switch candidate. The source was retrieved in 2026; its
archive represents issued forecasts, but an individual run's publication
timestamp is not embedded in this stitched series. This is a remaining
availability limitation, recorded in the candidate evaluation.

Open-Meteo source data is attributed under [CC BY 4.0](https://open-meteo.com/en/terms).
The API endpoint's free use has [non-commercial terms](https://open-meteo.com/en/pricing).
