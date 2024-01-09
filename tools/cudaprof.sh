#!/bin/bash
set -e
make -q build/cudaprof || make build/cudaprof
if [ "$1" == "-s" ]; then
    shift
    export PROF_SYNC=1
fi
CUDA_INJECTION64_PATH=build/cudaprof "$@"
