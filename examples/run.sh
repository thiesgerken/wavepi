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

mkdir -p $1
cd $1
cp ../$1.cfg wavepi.cfg
../../build/wavepi --export-config -c wavepi.cfg > wavepi_exported.cfg
../../build/wavepi -c wavepi.cfg
cat wavepi.log | ../../build/wavepi_logfilter 2 > wavepi.2.log
cat wavepi.log | ../../build/wavepi_logfilter 4 | xz > wavepi.4.log.xz
rm wavepi.log

cd ..
