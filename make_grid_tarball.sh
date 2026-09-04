#!/bin/bash
# make_grid_tarball.sh — build a relocatable lardon tarball for RCDS / justin-cvmfs-upload.
#
# Bundles, with NO conda-unpack (the env is relocated at runtime by setup.sh):
#   env/            <- the pixi env, packed by conda-pack (its own Python + xrootd + deps)
#   lardon/src/...  <- the lardon package (+ settings/)
#   setup.sh        <- runtime activation handler
#
# Usage (from anywhere):
#   bash /path/to/lardon/make_grid_tarball.sh [options]
#
# Options:
#   -e, --env NAME       pixi environment to pack            (default: grid)
#   -o, --output FILE    output tarball path                 (default: <repo>/lardon-<env>.tar)
#   -p, --pixi PATH      path to the pixi binary             (default: from PATH, else $PIXI_HOME/bin/pixi)
#   -s, --stage DIR      staging directory                   (default: <repo>/.pkg_<env>)
#   -k, --keep-staging   keep the staging dir (don't delete it afterward)
#   -h, --help           show this help and exit
#
# Notes:
#  - Requires pixi (on PATH or via --pixi). Installs the env if it isn't built yet.
#  - Run inside the TARGET OS (e.g. the SL7 container) so binaries match the grid nodes.

set -euo pipefail

# print the leading comment block (everything from line 2 up to the first non-# line)
usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]:-$0}"; }

# Repo root = the directory holding this script (alongside setup.sh / pixi.toml).
_SRC="${BASH_SOURCE[0]:-$0}"
PROJ="$(cd "$(dirname "$_SRC")" && pwd)"

# --- defaults ---
ENV_NAME="grid"
OUTPUT=""
PIXI=""
STAGE=""
KEEP_STAGING=0

# --- parse flags ---
while [ $# -gt 0 ]; do
  case "$1" in
    -e|--env)          shift; [ $# -ge 1 ] || { echo "ERROR: --env requires a value"    >&2; exit 2; }; ENV_NAME="$1" ;;
    --env=*)           ENV_NAME="${1#*=}" ;;
    -o|--output)       shift; [ $# -ge 1 ] || { echo "ERROR: --output requires a value" >&2; exit 2; }; OUTPUT="$1" ;;
    --output=*)        OUTPUT="${1#*=}" ;;
    -p|--pixi)         shift; [ $# -ge 1 ] || { echo "ERROR: --pixi requires a value"   >&2; exit 2; }; PIXI="$1" ;;
    --pixi=*)          PIXI="${1#*=}" ;;
    -s|--stage)        shift; [ $# -ge 1 ] || { echo "ERROR: --stage requires a value"  >&2; exit 2; }; STAGE="$1" ;;
    --stage=*)         STAGE="${1#*=}" ;;
    -k|--keep-staging) KEEP_STAGING=1 ;;
    -h|--help)         usage; exit 0 ;;
    --)                shift; break ;;
    -*)                echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)                 echo "ERROR: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# --- apply derived defaults ---
[ -n "$OUTPUT" ] || OUTPUT="$PROJ/lardon-${ENV_NAME}.tar"
[ -n "$STAGE" ]  || STAGE="$PROJ/.pkg_${ENV_NAME}"

# Locate pixi.
if [ -z "$PIXI" ]; then
  PIXI="$(command -v pixi || true)"
  if [ -z "$PIXI" ] && [ -x "${PIXI_HOME:-$HOME/.pixi}/bin/pixi" ]; then
    PIXI="${PIXI_HOME:-$HOME/.pixi}/bin/pixi"
  fi
fi
if [ -z "$PIXI" ] || [ ! -x "$PIXI" ]; then
  echo "ERROR: pixi not found. Put it on PATH or pass --pixi /path/to/pixi." >&2
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

echo "[5/5] writing tarball: $OUTPUT"
tar -C "$STAGE" -czf "$OUTPUT" .
[ "$KEEP_STAGING" = "1" ] || rm -rf "$STAGE"

echo
echo "Done."
ls -lh "$OUTPUT"
command -v sha1sum >/dev/null 2>&1 && echo "sha1: $(sha1sum "$OUTPUT" | awk '{print $1}')"
echo
echo "Next:"
echo "  justin-cvmfs-upload $OUTPUT      # from an RCDS-reachable node; prints INPUT_TAR_DIR_LOCAL"
echo "  source \$INPUT_TAR_DIR_LOCAL/setup.sh   # in the job, then run lardon-run ..."
