#!/bin/bash

cmdline="python -u main.py --phase validate --layer-dependent --analysis kobj"

DATA_DIR="train/validate/kobj/"

if [ $# -lt 2 ]
then
    echo "Usage: validate.sh dataset model [model ...]";
    exit -1;
fi

dataset="$1";

if [ -e "bin/$dataset" ]
then
    cmdline="$cmdline --dumped-graphs bin/$dataset";
else
    cmdline="$cmdline --graphs $(ls -d $DATA_DIR/$dataset-*) --dump-graphs-to bin/$dataset";
fi;

cmdline="$cmdline --validate-models ${@:2}";

echo "$cmdline";
$cmdline;
