#!/bin/bash

echo "deb http://deb.debian.org/debian jessie-backports main contrib" >> /etc/apt/sources.list
apt-get update && apt-get upgrade -y
apt-get install -t jessie-backports cmake make gcc g++ curl git libgtest-dev -y
apt-get install libopenmpi-dev libtbb-dev openmpi-bin -y

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

echo "** Building & Installing Boost 1.62 **"
cd /root
curl https://netcologne.dl.sourceforge.net/project/boost/boost/1.62.0/boost_1_62_0.tar.gz -O
tar xfz boost_1_62_0.tar.gz
cd boost_1_62_0
# deal.II must not find boost, otherwise it will try to link dynamically to it.
# (therefore putting it in local, and build deal.II first)
./bootstrap.sh --prefix=/usr/local
./b2 -j2
./b2 install

echo "** Building & Installing GTest **"
cd /root
mkdir gtest
cd gtest
cmake /usr/src/gtest
make -j2
cp libgtest.a libgtest_main.a /usr/lib/

echo "** Cleaning up **"
cd /root
rm -Rf boost_1_62_0 boost_1_62_0.tar.gz
rm -Rf dealii
rm -Rf gtest
