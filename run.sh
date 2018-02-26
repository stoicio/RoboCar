#!/bin/bash

## Two Birds, One Stone
# 1. sources conda environment
# 2. prevents the zombie container issue when started as pid 1

set -e
. activate carnd-term1

if [ -z "$1" ]
  then
    jupyter notebook --allow-root
elif [ "$1" == *".ipynb"* ]
  then
    jupyter notebook "$1" --allow-root
else
    exec "$@"
fi
