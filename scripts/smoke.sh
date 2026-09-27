#!/usr/bin/env sh
set -eu

base_url="${BASE_URL:-http://localhost:8000}"
frontend_url="${FRONTEND_URL:-http://localhost:8080}"

curl --fail --silent --show-error "${base_url}/api/v1/health/ready" | grep -q '"status":"ready"'
curl --fail --silent --show-error "${base_url}/api/v1/routes" | grep -q '"number":"1"'
curl --fail --silent --show-error "${base_url}/api/v1/forecasts?route_id=1001&horizon=day" | grep -q '"horizon":"day"'
curl --fail --silent --show-error "${base_url}/api/v1/routes" | grep -q '"number":"5"'
curl --fail --silent --show-error "${base_url}/api/v1/forecasts?route_id=1005&horizon=day" | grep -q '"target":"validation_count"'
curl --fail --silent --show-error "${base_url}/api/v1/forecasts?route_id=1005&horizon=month" | grep -q '"horizon":"month"'
curl --fail --silent --show-error "${frontend_url}" | grep -q 'Тормоза'
curl --fail --silent --show-error "${frontend_url}/api/v1/health/ready" | grep -q '"status":"ready"'
curl --fail --silent --show-error -X POST -H 'Content-Type: application/json' \
  -d '{"route_id":1,"horizon":"day","additional_vehicles":2,"interval_change_percent":-10,"demand_change_percent":20}' \
  "${frontend_url}/api/v1/scenarios/evaluate" | grep -q '"solver_version":"network-flow-baseline-v1"'

echo "Smoke checks passed: database, API, forecast and frontend are reachable."
