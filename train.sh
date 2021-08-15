#!/bin/bash

cmdline="python -u main.py --phase pretrain --layer-dependent --analysis kobj --latent-dim 64 --lr 1e-3 $@";

GRAPH_BIN="bin/hedc";
GRAPH_RAW_DIR="train/kobj/hedc";

if [ -e $GRAPH_BIN ]
then
    cmdline="$cmdline --dumped-graphs $GRAPH_BIN";
else
    if [ -d "$GRAPH_RAW_DIR-0" ]
    then
        cmdline="$cmdline --graphs $(ls -d $GRAPH_RAW_DIR-*) --dump-graphs-to $GRAPH_BIN";
    else
        false;
    fi;
fi;

echo "$cmdline";
$cmdline;
