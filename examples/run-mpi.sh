#!/bin/bash

mpirun --bind-to none -np 2 --hostfile hostfile ./run.sh $1
