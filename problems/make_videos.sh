#!/bin/bash

for f in $(find $1 -wholename "*$2*"); do
  if [ "${f##*.}" == "pvd" ]; then
    if [[ ! -f ${f%.pvd}.ogv ]]; then
      echo $f
      ./video2d.py $f $3 $4 $5 $6
    else
      echo "skipping $f"
    fi
  fi
done
