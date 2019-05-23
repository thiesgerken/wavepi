#! /bin/bash

docker build . -t git.thiesgerken.de:5005/thies/wavepi/dealii
docker push git.thiesgerken.de:5005/thies/wavepi/dealii
