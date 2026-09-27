# PostgreSQL repository tests

`make backend-sql-test` requires Docker Compose and the locked backend toolchain
(`make sync-backend`). It upgrades an empty PostgreSQL 17 database through Alembic,
checks schema drift, runs this directory, and removes its own project and volume.
`make verify` and the backend CI job include this gate. Plain `make backend-check`
skips these database tests explicitly and never connects to the developer DB.

Use `sql_session` for repository tests. Its external transaction rolls back even
when a repository commits its session. Use `sql_engine` with `isolated_session`
when a test needs to prove behavior across transaction contexts. Seed rows come
from the existing migrations; fixtures do not use `create_all` or truncate shared
tables. New tests must use explicit unused IDs because the current seed inserts
fixed IDs without advancing PostgreSQL sequences.

The runner accepts no external database URL or project name. Each invocation owns
a UUID project, project-scoped volume and Docker-assigned loopback ports. Inherited
Compose settings and `.env` do not select test resources. `migration-verify` and
`stack-verify` share this lifecycle; the latter runs the existing HTTP/frontend
smoke against its own published ports. Never substitute a working DB connection.

Failure exits remain nonzero, including cleanup failure. SIGINT/SIGTERM attempt
cleanup; SIGKILL, machine failure or an unavailable Docker daemon cannot guarantee
it. The first output line prints the owned project; after checking its label and
resources, recovery is `docker compose --env-file /dev/null -f compose.yaml
--project-name <printed-project> down --volumes --remove-orphans`. Do not remove
another invocation's project. Built images remain in the normal Docker cache.

These tests certify current synthetic seed behavior only. Multi-run selection,
stop/time-window filtering, nullable-key uniqueness and calibrated forecast
quality are separate contracts for later serving/model tasks.
