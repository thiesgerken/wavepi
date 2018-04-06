#!/bin/bash

for f in $(find $1 -name "reconstruction.ogv"); do
      DEST="$(dirname $f)/../../$(basename $(dirname $(dirname $f))).ogv"

      if [[ ! -f $DEST ]]; then
        echo "$f -> $DEST"

        cp $f $DEST
      fi

      DEST="$(dirname $f)/../$(basename $(dirname $f)).ogv"

      if [[ ! -f $DEST ]]; then
        echo "$f -> $DEST"

        cp $f $DEST
      fi
done

for f in $(find $1 -name "estimate.ogv"); do
      DEST="$(dirname $f)/../$(basename $(dirname $f)).ogv"

      if [[ ! -f $DEST ]]; then
        echo "$f -> $DEST"

        cp $f $DEST
      fi
done
