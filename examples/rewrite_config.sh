#!/bin/bash

echo $1

if [[ ! -f $1.cfg ]]; then
 echo "$1.cfg not found"
 exit 1
fi

../build/wavepi --export-config -c $1.cfg > $1.cfg.new
cp $1.cfg.new $1.cfg
rm $1.cfg.new
