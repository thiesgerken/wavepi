#!/bin/bash

#
# wavepi executable has to be in path!

NAME=${1%%.*}
echo $NAME

if [[ ! -f $1 ]]; then
 echo "$1 not found"
 exit 1
fi

if [[ -d $NAME ]]; then
 echo "directory $NAME exists"
 exit 2
fi

mkdir -p $NAME
cd $NAME
cp ../$1 $1

cp $1 wavepi.cfg
wavepi --export Diff -c $1 > wavepi-diff.cfg
wavepi --export -c $1 > wavepi-full.cfg

wavepi -c $1

cat wavepi.log | wavepi-logfilter 2 > wavepi.2.log
cat wavepi.log | wavepi-logfilter 100 | xz > wavepi.log.xz
rm wavepi.log
cd ..
