# Historical schedule recovery: public archive lane

Checked 2026-09-27. No usable January–March 2025 tram trip timetable was
recovered in this bounded search. This is a measured search outcome, **not**
a claim that historical schedules do not exist. No feed validity date,
mapping result or stop training data was modified. Primary transport-site
live date/API research and existing GTFS calendar analysis belong to the
other task lanes.

## Actual archived response recovered

Common Crawl `CC-MAIN-2025-08` returned one matching official schedule URL:
[route path 141503930](https://transport.mos.ru/transport/schedule/route/141503930),
captured **2025-02-11T12:58:48Z** with HTTP 200. Exact compressed WARC range
535644659–535664212 was downloaded successfully with HTTP 206 from:

[Common Crawl WARC](https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-08/segments/1738831951706.87/warc/CC-MAIN-20250211103744-20250211133744-00826.warc.gz)

Decompression produced 151,069 bytes. The payload contains the official
site header, navigation, footer and survey, but **no route heading, schedule
form or dated stop timetable**. The main content container is empty.
It cannot identify a target tram, departure time, direction or duty.
The HTTP 200 status alone is not schedule evidence. Exact index record,
response hashes and marker counts are in `receipts.json`.

## Other source families checked

| Source | Observed result | Why it cannot replace Q1 schedules |
|---|---|---|
| [MobilityDatabase mdb-3226](https://mobilitydatabase.org/feeds/gtfs/mdb-3226) | The full page says all datasets loaded; only `mdb-3226-202606301640` appears in its HTML/history | No earlier snapshot listed; later snapshot coverage must be established per service, not assumed |
| [Wayback CDX](https://web.archive.org/cdx/search/cdx) | Queries for transport.mos.ru schedule paths, rasp.mosgortrans.ru, producer GTFS URL returned HTTP 503 or timed out on retry | Search temporarily unavailable; no absence inference allowed |
| [Wayback availability API](https://archive.org/wayback/available) | Exact source/route URLs returned empty snapshot maps | Not reliable as exhaustive evidence: it also missed a known July snapshot |
| Wayback calendar API | Known old route path141501932 returns July29 HTTP200; 18 additional explicit path IDs mostly empty;420 has July30 HTTP404 | No Q1 capture surfaced in these exact-key queries; URL namespaces are not interchangeable |
| [Common Crawl index](https://index.commoncrawl.org/) | February source recovered as above; January/ March schedule-prefix attempts and retries gave 503/504/502; January oldhost query gave no captures | Only recovered payload is an empty page; remaining searches incomplete due upstream errors |
| [mosgortrans-schedule-history](https://github.com/mrKPbIS/mosgortrans-schedule-history) | Tree contains parser and test HTML; newest commit2020-06-14 | Not a2025 archive, despite repository name |
| [Elisei564/Mosgortrans](https://github.com/Elisei564/Mosgortrans) | Only July2025 dataset3221 and60662 CSVs | Stops/reference data, no stop_times/trips/calendar archive |
| [moscow-transit-bot](https://github.com/Rowlyge/moscow-transit-bot) | API client and database code, no cached schedule files; client uses apidata.mos.ru/v1 with api_key | Helpful retrieval implementation, no historical snapshot to ingest |
| [Transitland atlas](https://github.com/transitland/transitland-atlas) | Public repository tree inspected; no Moscow/Russia-named feed path identified | A tree-name search is not proof of no coverage; no suitable feed obtained |
| [KSUPT downloader repository](https://github.com/MBKCZAR/KSUPT-FileDownloader-Releases) | Describes downloader for operational ksupt.sldv.mosgortrans.com | No publicQ1 schedule artifact identified; operational access was not attempted |

The forum discussion is only a discovery lead, not authoritative schedule
truth. [Page221](https://forumot.ru/topic/30179-изменения-в-режимах-маршрутов-и-их-расписаниях/page/221/)
has March2024 links to official datasets: **60661 stop times**,60662 stops,
60664 routes,60665 trips,60666 calendars. Dataset60663 is transport operators,
not stop times. No Q1 export was attached in the inspected material.
Page250 was downloaded and dates to October2025; no CSV/ZIP/XLSX attachment
links were found. Current public [BusMaps dataset page](https://busmaps.com/en/russia/open-data-portal-moscow/moscow-official)
requires manual data delivery and shows a2026 build, not a downloadableQ1
release. No messages or data requests were sent to third parties.

## Boundaries and continuation

Current official-site route path IDs, GTFS route IDs and legacy141… IDs are
separate namespaces. The additional Wayback path probes are exploratory URL
queries, not identity joins. Do not manufacture old schedule validity by
redating a current feed. A stop-departure HTML table without trip/duty IDs
would require a separately labelled ambiguity-aware inference experiment;
it is not interchangeable with the duty-clock mapper.

Next useful evidence: an actual pre-April2025 export of datasets60661,
60665 and60666 (plus its route/stops dictionaries), or a verified archived
GTFS with matching validity and duty/trip IDs. Retry the failed archive
indexes when available; absence was not established here. A source's
publication/capture date and service applicability must both be retained.

Local raw research cache and scripts:
`/tmp/tram-historical-schedules-research`. Downloaded remote code was read,
not executed; only locally authored bounded fetch scripts ran. Public HTTP
response receipts/hashes are committed, full HTML/WARC stays outside Git.
Python TLS trust-store failure was resolved by using system curl verification;
TLS verification was not disabled. No repository implementation changed.

Validation: actual HTTP/index/range responses inspected; recovered WARC gzip
successfully decompressed; JSON receipts parsed; `git diff --check` passed.
No full application test suite is needed for this read-only research report.
