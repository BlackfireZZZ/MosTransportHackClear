# Deploying the Overpass API

How the instance described in [overpass-api.md](overpass-api.md) is built, and how
to rebuild it. Host: `2.29.6.201`, Ubuntu, **8 vCPU / 15 GB RAM**, 38 GB root
disk plus a 40 GB attached volume.

The host was 2 vCPU and 3.8 GB when this was written, and the settings below were
forced by that. They are kept because they still cost nothing, but read them as a
floor that works on a small box, not as tuning for this one.

## Layout

```
/srv/overpass/stack/          symlink -> /mnt/HC_Volume_106906767/overpass/stack
  docker-compose.yml          tracked here as infra/overpass/docker-compose.yml

/mnt/HC_Volume_106906767/overpass/
  db/
    db/                       Overpass database, ~11 GB
    planet_src.osm.bz2        converted source kept for re-imports, ~1.5 GB
    replicate_id              Geofabrik sequence number of the last applied diff
```

Data lives on the volume, not the root disk, so the instance survives rebuilding
the host. The volume is in `/etc/fstab` with `nofail` — a detached volume must not
block boot.

## From a bare host

**1. Docker**

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -qq
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
```

**2. Volume**

Attach it, then mount by stable id rather than `/dev/sdX`, which is not stable
across reboots:

```bash
echo "/dev/disk/by-id/scsi-0HC_Volume_<id> /mnt/HC_Volume_<id> ext4 discard,nofail,defaults 0 0" \
  >> /etc/fstab
mount -a
mkdir -p /mnt/HC_Volume_<id>/overpass/{db,stack}
ln -sfn /mnt/HC_Volume_<id>/overpass /srv/overpass
```

**3. Swap**

8 GB on the root disk — **not** on the volume, which is network-attached storage
and far too slow to swap to. This is insurance against an import spike, not extra
capacity; if the importer ever genuinely lives in swap, the job is finished either
way.

```bash
fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo "/swapfile none swap sw 0 0" >> /etc/fstab
echo "vm.swappiness=10" > /etc/sysctl.d/99-swappiness.conf
```

**4. Firewall**

Allow SSH *before* enabling, or you lock yourself out.

```bash
ufw default deny incoming && ufw default allow outgoing
ufw allow 22/tcp && ufw allow 80/tcp
ufw --force enable
```

Note that Docker publishes ports through its own iptables chain and bypasses ufw,
so port 80 would be reachable regardless. That is the intent here, but do not rely
on ufw to close a published container port.

**5. Source data**

Geofabrik publishes only `.pbf`, while the image's `init_osm3s.sh` pipes the planet
file through `bunzip2`. Feeding it a `.pbf` fails with `bzip2 error: read failed: -5`
and, because the container restarts, re-downloads 837 MB each time round. Convert
once and keep the result:

```bash
cd /mnt/HC_Volume_<id>/overpass/db
curl -L -o planet.osm.pbf \
  https://download.geofabrik.de/russia/central-fed-district-latest.osm.pbf
docker run --rm -v "$PWD:/w" -w /w --entrypoint osmium wiktorn/overpass-api:latest \
  cat -o planet_src.osm.bz2 planet.osm.pbf     # ~25 min, single-threaded
rm planet.osm.pbf
```

`OVERPASS_PLANET_URL` then points at `file:///db/planet_src.osm.bz2`, so re-imports
skip both the download and the conversion.

**6. Start**

```bash
cp infra/overpass/docker-compose.yml /srv/overpass/stack/
cd /srv/overpass/stack && docker compose up -d
```

About 50 minutes: import, then diff catch-up, then the area index. The API starts
answering when `/db/init_done` appears.

## Re-import without re-downloading

`init_done` gates the whole import block in the entrypoint, so restarting a
container that has it never re-imports — safe for config changes. To force a
rebuild:

```bash
cd /srv/overpass/stack && docker compose down
D=/mnt/HC_Volume_<id>/overpass/db
rm -rf $D/db $D/init_done $D/replicate_id $D/diffs $D/changes.log   # keep planet_src.osm.bz2
docker compose up -d
```

## Settings that matter on this host

Changing any of these back to the image default will break something that is not
obvious from the symptom.

| Setting | Value | Why not the default |
|---|---|---|
| `OVERPASS_FLUSH_SIZE` | `1` | Import buffer in GiB. The default of `16` drove `update_database` to 3.3 GB RSS and 4 GB of swap within two minutes of starting, heading for an OOM kill. At `1` the importer stays inside real RAM and runs no slower — the bottleneck is CPU on decompression, not the buffer. |
| `OVERPASS_TIME` | `3600` | The dispatcher-wide **pool** of time units shared by all queries, not a per-query limit. At the default of `180`, a single `[timeout:170]` query claimed nearly the whole pool and was load-shed, while small queries kept working — which looks like random flakiness. |
| `OVERPASS_ALLOW_DUPLICATE_QUERIES` | `yes` | Parsed as `(yes\|no)`. Any other value, `true` included, silently leaves duplicate blocking **on**. The container healthcheck repeats one fixed query, so it stays red forever while the API itself works. |
| `OVERPASS_META` | `yes` | Keeps version, timestamp and user per object. `attic` would add full history and multiply the database size several times over; there is no room for that here. |
| `OVERPASS_SPACE` | 2 GiB | Shared pool. Bounds real memory use across the 4 concurrent queries the rate limit allows. |

## Health

```bash
docker inspect -f '{{.State.Health.Status}}' overpass
docker exec overpass /app/bin/dispatcher --status --db-dir=/db/db
cat /mnt/HC_Volume_<id>/overpass/db/replicate_id     # last applied Geofabrik sequence
```

- A rising `load shedded requests` means queries are claiming more of the time or
  space pool than is free — raise `OVERPASS_TIME` or lower the `[timeout:]` clients
  ask for.
- After a restart, `rules_loop` rebuilds the area index and briefly blocks readers.
  Dispatcher timeouts in that window are expected and clear on their own.

## Not done

- **No TLS and no authentication.** The API is open to the internet over plain HTTP.
  Acceptable for a hackathon on a throwaway host; not acceptable for anything
  holding real traffic. With a domain, put Caddy in front for automatic
  Let's Encrypt.
- **Concurrency is 4, server-wide.** A team that parallelises hard will shed each
  other's queries. Cache locally instead — which is what the extraction scripts do.
