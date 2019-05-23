#!/bin/bash

echo "** Building & Installing Boost 1.62 **"
cd /root
git clone https://github.com/boostorg/boost.git
cd boost
git checkout boost-1.62.0
./bootstrap.sh --prefix=/usr
# deal.II must not find shared boost library, otherwise it will try to link dynamically to it!
./b2 cxxflags=-fPIC cflags=-fPIC -j16 variant=release link=static threading=multi runtime-link=static install

echo "** Cleaning up **"
cd /root
rm -Rf boost