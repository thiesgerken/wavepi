#!/bin/bash

echo "** Building & Installing Boost **"
cd /root
# git clone https://github.com/boostorg/boost.git
# cd boost
# git checkout boost-1.62.0
# git submodule init
# git submodule update
# curl https://netcologne.dl.sourceforge.net/project/boost/boost/1.62.0/boost_1_62_0.tar.gz -O
curl https://dl.bintray.com/boostorg/release/1.63.0/source/boost_1_63_0.tar.gz -OL
tar xfz boost_1_63_0.tar.gz
cd boost_1_63_0
./bootstrap.sh --prefix=/usr
# deal.II must not find shared boost library, otherwise it will try to link dynamically to it!
./b2 cxxflags=-fPIC cflags=-fPIC -j16 variant=release link=static threading=multi runtime-link=static install

echo "** Cleaning up **"
cd /root
# rm -Rf boost
rm -Rf boost_1_63_0 boost_1_63_0.tar.gz