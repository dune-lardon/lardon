#!/bin/bash
# setup.sh — activate lardon's self-contained env at runtime, from EITHER:
#   (a) a packaged/relocated tarball (e.g. unpacked on read-only cvmfs), layout:
#         <dir>/setup.sh
#         <dir>/env/            <- the conda-pack'd conda prefix (its own Python)
#         <dir>/lardon/src/...  <- the lardon package (+ settings/)
#   (b) a normal repo checkout, layout:
#         <repo>/setup.sh
#         <repo>/src/lardon/... <- the lardon package (+ settings/)
#         <repo>/.pixi/envs/<env>/  <- a pixi-installed env (e.g. grid/default)
# Works from any path with NO file modification (no conda-unpack needed).
#
# Usage:  source setup.sh   then   lardon-run <lardon args...>
#   In a repo checkout you can pick which pixi env to use:
#     LARDON_PIXI_ENV=grid source setup.sh        # default search order: grid, default, prod, test
#   For remote root:// inputs, scope the xrootd preload to the single call:
#     LD_PRELOAD="$LARDON_XROOTD_PRELOAD_LIB" lardon-run -file root://... -det pdhd -trk

# 1. Locate this script (works when sourced under bash)
_SRC="${BASH_SOURCE[0]:-$0}"
HERE="$(cd "$(dirname "$_SRC")" && pwd)"

# 1b. Detect layout -> set ENVROOT (conda prefix) and LARDONSRC (parent of the `lardon` package).
ENVROOT=""
LARDONSRC=""
if [ -d "$HERE/env" ] && [ -d "$HERE/lardon/src/lardon" ]; then
  # (a) packaged tarball
  ENVROOT="$HERE/env"
  LARDONSRC="$HERE/lardon/src"
elif [ -d "$HERE/src/lardon" ]; then
  # (b) repo checkout: source is <repo>/src, env is <repo>/.pixi/envs/<env>
  LARDONSRC="$HERE/src"
  for _cand in "$LARDON_PIXI_ENV" grid default prod test; do
    [ -n "$_cand" ] && [ -d "$HERE/.pixi/envs/$_cand" ] && { ENVROOT="$HERE/.pixi/envs/$_cand"; break; }
  done
  # fall back to the first installed env if none of the preferred names exist
  if [ -z "$ENVROOT" ] && [ -d "$HERE/.pixi/envs" ]; then
    ENVROOT="$(ls -d "$HERE"/.pixi/envs/*/ 2>/dev/null | head -1)"; ENVROOT="${ENVROOT%/}"
  fi
fi

if [ -z "$LARDONSRC" ]; then
  echo "setup.sh: could not locate lardon source under $HERE (expected env/+lardon/src or src/lardon)" >&2
  return 1 2>/dev/null || exit 1
fi
if [ -z "$ENVROOT" ] || [ ! -x "$ENVROOT/bin/python" ]; then
  echo "setup.sh: could not find a usable env under $HERE" >&2
  echo "  tarball: expected $HERE/env ; repo: expected $HERE/.pixi/envs/<env> (run 'pixi install -e grid')" >&2
  return 1 2>/dev/null || exit 1
fi

# 2. Put the env's bin first so `python` IS the env's python (its shebangs are stale,
#    but the interpreter itself derives sys.prefix from its own location -> fine).
export PATH="$ENVROOT/bin:$PATH"
export CONDA_PREFIX="$ENVROOT"

# 3. Run package activation hooks if present (ca-certificates sets SSL_CERT_FILE to the
#    env's own bundle, gdal/proj set their data dirs, etc.) — this auto-fixes openssl.
if [ -d "$ENVROOT/etc/conda/activate.d" ]; then
  for _f in "$ENVROOT"/etc/conda/activate.d/*.sh; do
    [ -r "$_f" ] && . "$_f"
  done
fi

# 4. TLS belt-and-suspenders for openssl/xrootd (in case no activate.d hook ran).
if [ -z "$SSL_CERT_FILE" ] && [ -f "$ENVROOT/ssl/cacert.pem" ]; then
  export SSL_CERT_FILE="$ENVROOT/ssl/cacert.pem"
fi
# IGTF/grid CAs on cvmfs (used for X.509/xrootd auth):
export X509_CERT_DIR="${X509_CERTIFICATES:-/cvmfs/grid.cern.ch/etc/grid-security/certificates}"

# 5. lardon python + runtime paths.
#    PYTHONPATH makes `import lardon` work regardless of any stale editable .pth.
#    LARDON_PATH must point at the package dir that contains settings/.
export PYTHONPATH="$LARDONSRC${PYTHONPATH:+:$PYTHONPATH}"
export LARDON_PATH="$LARDONSRC/lardon"
# Outputs must go to a WRITABLE dir (cvmfs is read-only) — default to the job's cwd:
export LARDON_RECO="${LARDON_RECO:-$PWD/reco}"
export LARDON_PLOT="${LARDON_PLOT:-$PWD/plots}"
export QT_XCB_GL_INTEGRATION=none

# 6. xrootd POSIX preload — needed for lardon to open remote root:// inputs (h5py uses
#    plain POSIX I/O; this shim routes root:// paths to xrootd). We deliberately do NOT
#    export LD_PRELOAD globally: it would leak into every command in this shell and its
#    subprocesses and can break unrelated tools. Instead we expose the lib path, so you
#    scope it to a single call (the var reaches lardon-run's python child, then is gone):
#        LD_PRELOAD="$LARDON_XROOTD_PRELOAD_LIB" lardon-run -file root://... -det pdhd -trk
export LARDON_XROOTD_PRELOAD_LIB="$(ls "$ENVROOT"/lib/libXrdPosixPreload.so 2>/dev/null | head -1)"

# 7. Invoke lardon via -m to dodge the baked console-script shebang.
lardon-run() { python -m lardon.lardon "$@"; }
