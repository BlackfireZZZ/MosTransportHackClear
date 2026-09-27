# ADR-0007: explicit geometry quality

Accepted 2026-09-25, TASK-031.

## Contract

An absent edge polyline retains its endpoint connector for topology navigation, but
is `inferred`, never surveyed rail geometry. Supplied polylines are `provided`;
that label describes provenance, not a certification of accuracy. Explicit
`synthetic: true` in either graph or GeoJSON metadata labels all supplied
geometry `synthetic`; absent or false metadata in the other file cannot erase
that marker.
A path inherits inferred quality if any traversed edge is missing; otherwise it
inherits synthetic or provided quality. Missing-edge counts accompany paths and
GeoJSON metadata. Unreachable paths have no geometry and zero missing edges.

The map omits inferred connectors from its rail layers and suppresses a path
polyline containing inferred segments. Persistent text explains this omission;
accessible edge lists label quality. Synthetic geometry is explicitly labeled.
This preserves route reachability, distances, coordinate arrays and error codes.

## Evidence and alternatives

`test_geometry_falls_back_to_the_straight_line` freezes endpoint fallback, while
`test_missing_geometry_file_is_an_error_not_straight_lines` freezes failure for a
missing file. Rejecting every incomplete edge would discard usable topology and
break the former contract. Additive quality fields preserve navigation and make
this previously silent condition observable. Existing GeoJSON properties and
existing status text require no new algorithm or UX mechanism/dependency.

## Compatibility and rollback

Older clients ignore additive fields and retain their prior limitations. Deploy
server and current UI together. Rollback reverts this task's commit, restoring
prior fallback behavior; it does not mutate stored graph artifacts. Restore only
complete geometry artifacts if reverting UI quality handling in production.
