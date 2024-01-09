#!/bin/sh
set -e
if ! test -f build/cudaprof; then
    make build/cudaprof
fi
CUDA_INJECTION64_PATH=build/cudaprof "$@"
