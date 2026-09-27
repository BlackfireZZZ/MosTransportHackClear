# ADR-0004: immutable graph snapshots and an atomic manifest

Accepted 2026-09-23 for TASK-030. Extends ADR-0003's file repository; no public
graph API or runtime dependency changes.

## Contract

The extractor stages all five formats in the destination filesystem, validates
their structure, shared metadata and equivalent graph values, and hashes their
bytes. A schema-v1 manifest binds filenames, byte sizes, SHA-256 hashes and source
metadata. Its canonical body determines the snapshot version. Published files
live in `graph-store/snapshots/<version>/` and must never be edited in place.

Only after validation and file/directory synchronization does the publisher
replace `graph-store/tram_graph.manifest.json`. This is the commit point for an
existing store. First publication stages the entire store including this manifest
and atomically renames it to `graph-store/`; a failure before that rename leaves
the legacy pair available. The active manifest contains the same complete
manifest as its snapshot, not five
independently mutable paths. Readers capture it once and verify the bytes from
that snapshot, so a concurrent activation cannot mix JSON and GeoJSON versions.
Concurrent publishers may both succeed; the last manifest switch wins, always
selecting a complete set. No automatic pruning is performed.

The repository's existing same-directory `tram_graph.json`/`tram_graph.geojson`
configuration discovers the store manifest. Explicit alternate filenames retain
legacy pair loading. A missing manifest in an installed `graph-store/` is an
error, not a fallback to old flat files. Legacy committed files continue to work
when there is no versioned store. The API still caches one network per process;
activation affects new readers/processes, not cached in-flight snapshots.

Rollback validates an existing snapshot and atomically activates its manifest.
It does not overwrite files or delete later snapshots. HTTP response schemas,
path semantics and the existing per-edge geometry fallback stay unchanged.

## Research and limitations

Reviewed official sources on 2026-09-23:

- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
  specifies atomic successful rename and possible failure across filesystems.
  Staging and the manifest temporary file therefore live under the destination.
- [Python os.fsync](https://docs.python.org/3/library/os.html#os.fsync)
  describes flushing file descriptors; files are flushed before activation.
- [SQLite atomic commit](https://www.sqlite.org/atomiccommit.html) explains why
  ordered flushes and a single commit event matter and why broken filesystem
  guarantees undermine crash consistency. We reuse that ordering principle,
  not SQLite's rollback-journal algorithm: these small graph files are immutable,
  and only the manifest changes. There are no mutable database pages to undo.

Supported publication environment is a local POSIX filesystem with atomic rename
and directory fsync. Injected exceptions/process interruption before the manifest
switch leave the prior snapshot active; abandoned staging directories or sealed
unselected snapshots may remain after a killed process. An error during the final
directory fsync is an uncertain durability outcome after a successful switch:
inspect the active manifest before retrying. We do not claim verified power-loss,
network-filesystem, hostile-writer or Windows durability. Checksums detect accidental
corruption, not authenticity. Deployment keeps this data read-only to the API.

## Verification / recovery

Tests inject failures in each format writer, mismatch paired metadata and values,
tamper bytes and manifests, switch versions while reading, and roll back. A
round-trip of the committed set preserves 856 stops, 919 directed edges, and
components of 694 and 162. `make backend-check` and `make check` remain gates.
To reverse this format change, retain the old complete flat set and use an earlier
application/extractor revision; do not delete a manifest to force fallback inside
a versioned store.
