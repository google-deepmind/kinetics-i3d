#!/bin/bash

# Script to extract the TV-L1 flow from the HMDB-51 data set.
#
# usage:
#  bash hmdb_flow.sh [num threads]
NUM_THREADS=${2}

python hmdb_extract_flow.py \
 --num_threads=${NUM_THREADS}
