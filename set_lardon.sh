#!/bin/bash


export QT_XCB_GL_INTEGRATION=none
#eval "$(pixi shell-hook)"

#uncomment next two lines to open files outside of CERN/FERMILAB
#xrootd=$(find .pixi -name "libXrdPosixPreload.so")
#export LD_PRELOAD=$PWD/$xrootd

export LARDON_PATH=$PWD/src/lardon
export LARDON_RECO=$PWD/reco
export LARDON_PLOT=$PWD/plots
