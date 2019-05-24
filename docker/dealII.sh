#!/bin/bash

echo "** Building & Installing deal.II **"
cd /root
git clone https://github.com/dealii/dealii.git
cd dealii
git checkout v9.1.0
mkdir build
cd build
cmake .. -DDEAL_II_WITH_MPI=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DDEAL_II_COMPONENT_EXAMPLES=OFF
make -j16
make install
# keep a copy of the configuration log
cp detailed.log /root/dealii.log

echo "** Cleaning up **"
rm -Rf /root/dealii
