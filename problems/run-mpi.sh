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
wavepi --export -c $1 > wavepi.cfg

mpirun --bind-to none -np 2 $(which wavepi) -c $1

for i in {0..9}; do
  if [[ -f wavepi.log$i ]]; then
    cat wavepi.log$i | wavepi-logfilter 2 > wavepi.2.log$i
    cat wavepi.log$i | wavepi-logfilter 100 | xz > wavepi.log$i.xz
    rm wavepi.log$i
  fi
done

cd ..
