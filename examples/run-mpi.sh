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

mpirun --bind-to none -np 2 --hostfile hostfile mkdir -p $1

cd $1
cp ../$1.cfg wavepi.cfg
../../build/wavepi --export-config -c wavepi.cfg > wavepi_exported.cfg

mpirun --bind-to none -np 2 --hostfile hostfile ../../build/wavepi -c wavepi.cfg

cat wavepi.log | ../../build/wavepi_logfilter 2 > wavepi.2.log
cat wavepi.log | ../../build/wavepi_logfilter 4 | xz > wavepi.4.log.xz
rm wavepi.log
cd ..
