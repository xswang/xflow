#!/bin/bash
# set -x
if [ $# -gt 1 ]; then
    echo "usage: $0"
    exit -1;
fi

bin=$1
shift
arg="$@"

# start the scheduler
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8000
export DMLC_ROLE='scheduler'
${bin} ${arg} &

wait
