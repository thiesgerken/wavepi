#!/bin/bash

echo "** Building & Installing Boost 1.62 **"
cd /root
curl https://netcologne.dl.sourceforge.net/project/boost/boost/1.62.0/boost_1_62_0.tar.gz -O
tar xfz boost_1_62_0.tar.gz
cd boost_1_62_0
./bootstrap.sh --prefix=/usr
./b2 cxxflags=-fPIC cflags=-fPIC -j2 variant=release link=static threading=multi runtime-link=static install

echo "** Cleaning up **"
cd /root
rm -Rf boost_1_62_0 boost_1_62_0.tar.gz