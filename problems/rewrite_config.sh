#!/bin/bash

echo $1

if [[ ! -f $1 ]]; then
 echo "$1 not found"
 exit 1
fi

../build/wavepi --export-config -c $1 > $1.new
cp $1.new $1
rm $1.new
