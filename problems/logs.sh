#!/bin/bash
for i in {0..9}; do
  if [[ -f wavepi.log$i ]]; then
    cat wavepi.log$i | wavepi-logfilter 2 > wavepi.2.log$i
    cat wavepi.log$i | wavepi-logfilter 100 | xz > wavepi.log$i.xz
    rm wavepi.log$i
  fi
done
