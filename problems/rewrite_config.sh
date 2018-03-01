#!/bin/bash

NAME=${1%%.*}
echo $NAME

if [[ ! -f $1 ]]; then
 echo "$1 not found"
 exit 1
fi

wavepi --export Diff -c $1 > $1.new
cp $1.new $1
rm $1.new
