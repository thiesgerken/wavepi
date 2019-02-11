#!/bin/bash

echo "** Building & Installing deal.II **"
cd /root
git clone https://github.com/dealii/dealii.git
cd dealii
git checkout d68d619f14dab0cc4225178165f4c50fd0dd0e21
mkdir build
cd build
cmake .. -DDEAL_II_WITH_MPI=ON -DCMAKE_INSTALL_PREFIX=/usr/local -DDEAL_II_COMPONENT_EXAMPLES=OFF
make -j2
make install
# keep a copy of the configuration log
cp detailed.log /root/dealii.log

echo "** Cleaning up **"
rm -Rf /root/dealii
