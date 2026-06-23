#!/bin/bash
# make_grid_tarball.sh — build a relocatable lardon tarball for RCDS / justin-cvmfs-upload.
#
# Bundles, with NO conda-unpack (the env is relocated at runtime by setup.sh):
#   env/            <- the pixi env, packed by conda-pack (its own Python + xrootd + deps)
#   lardon/src/...  <- the lardon package (+ settings/)
#   setup.sh        <- runtime activation handler
#
# One-liner (from anywhere):
#   bash /path/to/lardon/make_grid_tarball.sh [ENV_NAME] [OUTPUT_TARBALL]
#
# Defaults : ENV_NAME=grid   OUTPUT=<repo>/lardon-<ENV_NAME>.tar
# Overrides: PIXI=/path/to/pixi   STAGE=/path/to/staging   KEEP_STAGING=1
#
# Notes:
#  - Requires pixi on PATH (or $PIXI). Installs the env if it isn't built yet.
#  - Run inside the TARGET OS (e.g. the SL7 container) so binaries match the grid nodes.

set -euo pipefail

# Repo root = the directory holding this script (alongside setup.sh / pixi.toml).
_SRC="${BASH_SOURCE[0]:-$0}"
PROJ="$(cd "$(dirname "$_SRC")" && pwd)"

ENV_NAME="${1:-grid}"
OUT="${2:-$PROJ/lardon-${ENV_NAME}.tar}"
STAGE="${STAGE:-$PROJ/.pkg_${ENV_NAME}}"

# Locate pixi.
PIXI="${PIXI:-$(command -v pixi || true)}"
if [ -z "$PIXI" ] && [ -x "${PIXI_HOME:-$HOME/.pixi}/bin/pixi" ]; then
  PIXI="${PIXI_HOME:-$HOME/.pixi}/bin/pixi"
fi
if [ -z "$PIXI" ] || [ ! -x "$PIXI" ]; then
  echo "ERROR: pixi not found. Put it on PATH or set PIXI=/path/to/pixi." >&2
  exit 1
fi

# Sanity-check the repo.
[ -f "$PROJ/pixi.toml" ]  || { echo "ERROR: $PROJ/pixi.toml not found"  >&2; exit 1; }
[ -d "$PROJ/src/lardon" ] || { echo "ERROR: $PROJ/src/lardon not found" >&2; exit 1; }
[ -f "$PROJ/setup.sh" ]   || { echo "ERROR: $PROJ/setup.sh not found"   >&2; exit 1; }

echo "[1/5] pixi: $("$PIXI" --version)"

ENVDIR="$PROJ/.pixi/envs/$ENV_NAME"
if [ ! -x "$ENVDIR/bin/python" ]; then
  echo "[2/5] env '$ENV_NAME' not installed -> pixi install -e $ENV_NAME"
  ( cd "$PROJ" && "$PIXI" install -e "$ENV_NAME" )
else
  echo "[2/5] using existing env: $ENVDIR"
fi

echo "[3/5] conda-pack '$ENV_NAME' into staging ($STAGE)"
rm -rf "$STAGE"
mkdir -p "$STAGE/lardon/src"
# --format no-archive packs straight into a directory (no intermediate tarball);
# --ignore-editable-packages skips the editable 'lardon' (we ship its source below).
# We deliberately do NOT run conda-unpack: setup.sh relocates at runtime.
"$PIXI" exec conda-pack -p "$ENVDIR" --ignore-editable-packages --format no-archive -o "$STAGE/env"

echo "[4/5] adding lardon source + setup.sh"
cp -r "$PROJ/src/lardon" "$STAGE/lardon/src/"
cp "$PROJ/setup.sh" "$STAGE/setup.sh"

echo "[5/5] writing tarball: $OUT"
tar -C "$STAGE" -czf "$OUT" .
[ "${KEEP_STAGING:-0}" = "1" ] || rm -rf "$STAGE"

echo
echo "Done."
ls -lh "$OUT"
command -v sha1sum >/dev/null 2>&1 && echo "sha1: $(sha1sum "$OUT" | awk '{print $1}')"
echo
echo "Next:"
echo "  justin-cvmfs-upload $OUT      # from an RCDS-reachable node; prints INPUT_TAR_DIR_LOCAL"
echo "  source \$INPUT_TAR_DIR_LOCAL/setup.sh   # in the job, then run lardon-run ..."
