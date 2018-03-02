#!/bin/bash

#
# wavepi executable has to be in path!

if [ "$1" == "-f" ]; then
  FILE=$2
  NAME=${FILE%%.*}

  if [[ ! -f $FILE ]]; then
   echo "$FILE not found"
   exit 1
  fi

  if [[ -d $NAME ]]; then
   echo "directory $NAME exists, deleting"
   rm -R $NAME
  fi
else
  FILE=$1
  NAME=${FILE%%.*}
  echo $NAME
fi

if [[ ! -f $FILE ]]; then
 echo "$FILE not found"
 exit 1
fi

if [[ -d $NAME ]]; then
 echo "directory $NAME exists"
 exit 2
fi

mkdir -p $NAME
cd $NAME
cp ../$FILE $FILE

wavepi --export Diff -c $FILE > wavepi-diff.cfg
wavepi --export -c $FILE > wavepi.cfg

wavepi -c $FILE

cat wavepi.log | wavepi-logfilter 2 > wavepi.2.log
cat wavepi.log | wavepi-logfilter 100 | xz > wavepi.log.xz
rm wavepi.log
cd ..
