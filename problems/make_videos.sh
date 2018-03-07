#!/bin/bash

for f in $(find . -wholename "*$1*"); do
  if [ "${f##*.}" == "pvd" ]; then
    if [[ ! -f ${f%.pvd}.ogv ]]; then
      echo $f
      ./video2d.py $f $2 $3 $4 $5 $6
    else
      echo "skipping $f"
    fi
  fi
done
