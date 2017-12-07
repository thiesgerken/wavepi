#!/bin/bash
for filename in *.cfg; do
  ./run.sh $(basename "$filename" .cfg) || true
done
