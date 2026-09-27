"""Collect bounded public traffic/news facts; raw HTML stays outside Git."""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, date, datetime
from pathlib import Path

from tramflow_ml.external_factors import moscow_publication_day, parse_traffic_page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, receipts = {}, []
    for before in range(19880, 23481, 20):
        url = f"https://t.me/s/DtOperativno?before={before}"
        path = args.cache / f"before-{before}.html"
        if not path.exists():
            if args.offline:
                raise ValueError(f"missing cached page {before}")
            response = subprocess.run(
                ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "30", url],
                check=True, capture_output=True,
            )
            path.write_bytes(response.stdout)
            time.sleep(.5)
        raw = path.read_bytes()
        receipts.append({"url": url, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        for row in parse_traffic_page(raw.decode()):
            day = moscow_publication_day(row["published_at"])
            if date(2025, 1, 1) <= day <= date(2025, 10, 31):
                rows[row["source_url"]] = row
    ordered = sorted(rows.values(), key=lambda row: row["published_at"])
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                      for row in ordered).encode()
    (args.output / "observations.jsonl").write_bytes(payload)
    traffic = [row for row in ordered if row["congestion_score"] is not None
               or row["mean_speed_kmh"] is not None]
    manifest = {"schema": "official-traffic-news.v1", "source": "https://t.me/s/DtOperativno",
                "publisher": "Moscow Department of Transport / CODD official public channel",
                "retrieved_at": datetime.now(UTC).isoformat(), "pages": receipts,
                "files": {"observations.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(),
                                                 "rows": len(ordered)}},
                "traffic_observations": len(traffic),
                "traffic_days": len({moscow_publication_day(row["published_at"]) for row in traffic}),
                "edited_traffic_observations": sum(row["edited"] for row in traffic),
                "timezone": "Europe/Moscow", "period": ["2025-01-01", "2025-10-31"],
                "actual_period": [str(moscow_publication_day(ordered[0]["published_at"])),
                                  str(moscow_publication_day(ordered[-1]["published_at"]))],
                "data_rights": "Public factual numbers, timestamps and classification flags only; "
                               "no article text or raw Telegram HTML redistributed. "
                               "No separate open-data licence located.",
                "limitations": ["Selective publication, not continuous hourly sensor coverage.",
                                "Citywide measurements, no route-specific speeds.",
                                "News flags count mentions, not unique incidents or active closures.",
                                "Edited posts excluded from features; original unedited vintage unproven.",
                                "Observation timestamp approximated by publication timestamp.",
                                "Missing publication never implies zero congestion or no incident."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({key: manifest[key] for key in ("files", "traffic_observations", "traffic_days")}))


if __name__ == "__main__":
    main()
