#!/bin/bash

cmdline="python -u main.py --phase pretrain --layer-dependent --analysis kobj --lr 1e-5";

GRAPH_BIN="bin/toba-s";
GRAPH_RAW_DIR="train/kobj/toba-s";

if [ -e $GRAPH_BIN ]
then
    cmdline="$cmdline --dumped-graphs $GRAPH_BIN";
else
    if [ -d "$GRAPH_RAW_DIR-0" ]
    then
        graphs="$cmdline --graphs $(ls -d $GRAPH_RAW_DIR-*) --dump-graphs-to $GRAPH_BIN";
    else
        false;
    fi;
fi;

echo "$cmdline";
$cmdline;
