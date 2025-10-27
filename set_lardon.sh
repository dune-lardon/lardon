#!/bin/bash

source .venv/bin/activate
xrootd=$(find .venv -name "libXrdPosixPreload.so")
export LD_PRELOAD=$PWD/$xrootd
export LARDON_PATH=$PWD/lardon
export LARDON_RECO=$PWD/reco
