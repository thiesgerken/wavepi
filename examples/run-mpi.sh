#!/bin/bash

# only check on root node
if [[ -d $1 ]]; then
 echo "directory $1 exists"
 exit 2
fi

mpirun --bind-to none -np 2 --hostfile hostfile rm -R $1
mpirun --bind-to none -np 2 --hostfile hostfile ./run.sh $1
