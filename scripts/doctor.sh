#!/usr/bin/env sh
set -eu

# Minimums, not pins. A newer runtime is fine; the tools below are asked to match
# ">= this", so a machine is never wrong for being ahead of the lockfile.
minimum_python="${PYTHON_VERSION:-3.13}"
minimum_node="${NODE_VERSION:-24}"
errors=0

pass() {
    printf 'ok  %s\n' "$1"
}

fail() {
    printf 'ERR %s\n' "$1" >&2
    errors=$((errors + 1))
}

require_command() {
    command_name="$1"
    if command -v "$command_name" >/dev/null 2>&1; then
        pass "$command_name is available"
    else
        fail "$command_name is missing"
    fi
}

for command_name in git uv node npm docker make; do
    require_command "$command_name"
done

if command -v uv >/dev/null 2>&1; then
    # No version arithmetic here: pyproject.toml declares `required-version`, so uv
    # refuses to run at all when it is too old. Reporting the version is enough.
    pass "uv is $(uv --version | awk '{print $2}')"

    # uv resolves the specifier itself, so ">=" needs no comparison in shell.
    if python_path="$(uv python find ">=$minimum_python" 2>/dev/null)"; then
        pass "Python $("$python_path" -c 'import platform; print(platform.python_version())') at $python_path"
    else
        fail "no Python >=$minimum_python (run: uv python install $minimum_python)"
    fi
fi

if command -v node >/dev/null 2>&1; then
    # A major version is a single integer, so this is plain shell arithmetic
    # rather than a version comparator.
    actual_node="$(node --version | sed 's/^v//' | cut -d. -f1)"
    if [ "$actual_node" -ge "$minimum_node" ]; then
        pass "Node major version is $actual_node (minimum $minimum_node)"
    else
        fail "Node major version is $actual_node; need >=$minimum_node"
    fi
fi

if command -v docker >/dev/null 2>&1; then
    if docker compose version >/dev/null 2>&1; then
        pass "Docker Compose is available"
    else
        fail "Docker Compose plugin is unavailable"
    fi
fi

for required_file in uv.lock frontend/package-lock.json compose.yaml .env.example; do
    if [ -f "$required_file" ]; then
        pass "$required_file exists"
    else
        fail "$required_file is missing"
    fi
done

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    if docker compose config --quiet; then
        pass "compose.yaml is valid"
    else
        fail "compose.yaml is invalid"
    fi
fi

if [ "$errors" -ne 0 ]; then
    printf '\nDoctor found %s problem(s). Use the dev container or install the pinned tools.\n' "$errors" >&2
    exit 1
fi

printf '\nDevelopment environment is ready.\n'
