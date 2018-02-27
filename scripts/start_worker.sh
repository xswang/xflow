#!/bin/bash
# set -x
if [ $# -gt 2 ]; then
    echo "usage: $0 num_workers bin [args..]"
    exit -1;
fi

export DMLC_NUM_WORKER=$1
shift
bin=$1
shift
arg="$@"

# start workers
export DMLC_ROLE='worker'
for ((i=0; i<${DMLC_NUM_WORKER}; ++i)); do
    export HEAPPROFILE=./W${i}
    ${bin} ${arg} &
done

wait
