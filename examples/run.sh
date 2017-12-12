#!/bin/bash

echo $1

if [[ ! -f $1.cfg ]]; then
 echo "$1.cfg not found"
 exit 1
fi

if [[ -d $1 ]]; then
 echo "directory $1 exists"
 exit 2
fi

mkdir $1
cd $1
cp ../$1.cfg wavepi.cfg
../../build/wavepi -c wavepi.cfg
cd ..
