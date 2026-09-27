# Overpass API

Self-hosted [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API) over
OpenStreetMap data for Moscow and Moscow oblast. Use it instead of the public
instances at `overpass-api.de` — those rate-limit aggressively and will throttle a
team hammering them from one address during a hackathon.

| | |
|---|---|
| Interpreter | `http://2.29.6.201/api/interpreter` |
| Status | `http://2.29.6.201/api/status` |
| Coverage | Central Federal District extract — Moscow and Moscow oblast in full |
| Freshness | Geofabrik daily diffs, polled hourly |
| Auth | none |
| Transport | HTTP only, no TLS |
| Licence | OpenStreetMap contributors, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/) — attribution required if you publish anything derived from it |

Deployment and operations live in [deployment.md](deployment.md).

## Making a request

The query goes in a `data` parameter, by GET or POST. Use POST for anything long —
some clients and proxies truncate long query strings.

```bash
curl -G http://2.29.6.201/api/interpreter --data-urlencode '
[out:json][timeout:60];
area["name"="Москва"]["admin_level"="4"]->.moscow;
node["station"="subway"](area.moscow);
out tags;
'
```

```python
import requests

OVERPASS = "http://2.29.6.201/api/interpreter"

def query(ql: str, timeout: int = 60) -> dict:
    r = requests.post(OVERPASS, data={"data": ql}, timeout=timeout + 30)
    r.raise_for_status()
    return r.json()   # raises on the HTML error page described below
```

The query language itself is Overpass QL; the
[QL reference](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL) and
[overpass turbo](https://overpass-turbo.eu/) (point it at this server under
Settings) are the places to learn it.

## Limits

These are the dispatcher settings on this instance, not Overpass defaults.

| Limit | Value | Meaning |
|---|---|---|
| `[timeout:]` ceiling | 300 s | largest value a single query may ask for |
| Time pool | 3600 units | shared across all in-flight queries |
| `[maxsize:]` default | 512 MiB | memory one query may claim |
| Space pool | 2 GiB | shared across all in-flight queries |
| Concurrency | 4 | queries running at once, server-wide |

The host has 2 vCPU and 3.8 GB RAM, so the ceiling is the whole team's, not each
person's. Two practical consequences:

- **Ask for the timeout you need, not the maximum.** `[timeout:]` is a claim against
  the shared pool for the duration of the query. Four people each asking
  `[timeout:300]` out of habit will shed each other's requests.
- **Filter by area or bbox before filtering by tag.** `node["highway"="bus_stop"]`
  unbounded scans the whole extract; bounded by a bbox it is near-instant.

## Errors

Overpass has five response shapes and only one is what a naive client expects.
All five were checked against this instance.

| Condition | HTTP | Content-Type | Body |
|---|---|---|---|
| Success | 200 | `application/json` | the JSON you asked for |
| Parse or static error — malformed QL | **400** | `text/html` | HTML error page |
| Dispatcher shed — see below | **504** | `text/html` | HTML error page |
| Query hit its own `[timeout:]`, under `[out:json]` | **200** | `application/json` | **valid JSON** with a `remark` key and possibly partial `elements` |
| Output is not JSON (no `[out:json]`) | 200 | `application/osm3s+xml` | XML |

Three things that catch clients out:

**A timed-out JSON query succeeds.** HTTP 200, valid JSON, `raise_for_status()`
passes, `.json()` parses — and the result quietly holds fewer elements than exist.
The only signal is a `remark` key:

```json
{"version": 0.6, "elements": [],
 "remark": "runtime error: Query timed out in \"query\" at line 1 after 2 seconds."}
```

Treating that as a complete answer is the single easiest way to ship wrong numbers.
Check for `remark` on every response.

**The HTML pages come with 400 and 504, not 200**, so `raise_for_status()` fires
before you reach the body and the message is lost. Read the body on failure.

**Asking for more than the server allows is reported as 504 "too busy"**, even when
the server is idle. `[timeout:]` above 300 or `[maxsize:]` above the 2 GiB pool is
shed with `Dispatcher_Client::request_read_and_idx::timeout`. It means the claim
was too large, not that the box is loaded.

The error text in an HTML page straddles a tag boundary — the word `Error` sits
inside a `<strong>`, the part that matters follows the closing tag:

```html
<p><strong style="color:#FF0000">Error</strong>: line 1: parse error: Unknown type "nodz" </p>
```

So a regex anchored on `error` and stopping at `<` returns the useless string
`"Error"`. Strip the tags first, then read the text. A client covering every shape:

```python
import html as html_mod
import json
import re

import requests

OVERPASS = "http://2.29.6.201/api/interpreter"

_BLOCK = re.compile(r"</(?:p|div|li|tr|h[1-6]|pre)\s*>|<br\s*/?>", re.I)
_TAG = re.compile(r"<[^>]*>")


def _page_errors(body: str) -> str:
    """Every error line on an Overpass HTML page, in order."""
    text = _TAG.sub("", _BLOCK.sub("\n", body))
    lines = [" ".join(l.split()) for l in html_mod.unescape(text).splitlines() if l.strip()]
    return "; ".join(l for l in lines if "error" in l.lower()) or " ".join(lines)[:300]


def query(ql: str, timeout: int = 60) -> dict:
    r = requests.post(OVERPASS, data={"data": ql}, timeout=timeout + 30)
    if r.status_code != 200:                      # 400 parse error, 504 shed
        raise RuntimeError(f"HTTP {r.status_code}: {_page_errors(r.text)}")
    if not r.text.lstrip().startswith("{"):       # HTML or XML under a 200
        raise RuntimeError(_page_errors(r.text))
    data = json.loads(r.text)
    if "remark" in data:                          # partial result, not a failure
        raise RuntimeError(f"incomplete: {data['remark']}")
    return data
```

Raising on `remark` is the safe default. Relax it only where a partial answer is
genuinely useful, and then surface the remark to the caller rather than dropping it.

A shed request is not a hard failure — retry it. Transient sheds for a minute or
two after a server restart are expected while the area index rebuilds.

`GET /api/status` reports the live picture — free slots, pool usage, and a
`load shedded requests` counter that rises when queries are being rejected.

## Recipes

Snippets below assume the area prelude:

```
area["name"="Москва"]["admin_level"="4"]->.moscow;
```

**Stops of one mode**

```
[out:json][timeout:60];
area["name"="Москва"]["admin_level"="4"]->.moscow;
node["public_transport"="stop_position"]["tram"="yes"](area.moscow);
out body;
```

**Routes with their members resolved** — relations, then the nodes and ways they
reference, then the geometry of those ways. This is the shape
[fetch_tram_graph.py](../scripts/fetch_tram_graph.py) uses.

```
[out:json][timeout:300];
area["name"="Москва"]["admin_level"="4"]->.moscow;
relation["type"="route"]["route"="tram"](area.moscow)->.routes;
.routes out body;
node(r.routes);
out body;
way(r.routes);
out body;
node(w);
out skel qt;
```

**Counting before downloading** — cheap way to size a query before running it for
real:

```
[out:json][timeout:60];
area["name"="Москва"]["admin_level"="4"]->.moscow;
relation["type"="route"]["route"~"^(bus|tram|trolleybus|subway)$"](area.moscow);
out count;
```

**A bbox instead of an area** — faster, and works without the area index:

```
node["highway"="bus_stop"](55.1,36.0,56.9,39.0);
```

Bounding box order is `(south, west, north, east)`. The one above covers Moscow
and the oblast.

## What the data looks like

Moscow public transport is mapped to the
[PTv2 scheme](https://wiki.openstreetmap.org/wiki/Public_transport), which is worth
knowing before writing queries against it:

- A route relation is **one direction** of one route. A there-and-back service is
  two relations, usually collected in a `type=route_master`.
- Members carry roles. `stop`, `stop_entry_only` and `stop_exit_only` are
  `stop_position` nodes **on the track itself**; `platform*` is where passengers
  wait, beside the track. The two are separate objects for the same stop.
- Members with an empty role are the track ways, in travel order. Individual ways
  may be stored reversed, so orientation has to be worked out by matching endpoints.

Because stop nodes sit on the track, distance between consecutive stops can be
measured along the real geometry instead of as a straight line — see
[tram-graph.md](tram-graph.md).

Counts as of the 2026-09-18 extract:

| | |
|---|---|
| Route relations in Moscow | 3387 (bus 2269, train 1006, tram 73, subway 34, trolleybus 5) |
| Bus stops and stop positions, Moscow + oblast | 49 696 |
| Metro stations, Moscow | 279 |
