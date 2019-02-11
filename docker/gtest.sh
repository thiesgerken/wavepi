#!/bin/bash

echo "** Building & Installing GTest **"
cd /root
mkdir gtest
cd gtest
cmake /usr/src/gtest
make -j2
cp libgtest.a libgtest_main.a /usr/lib/

echo "** Cleaning up **"
rm -Rf /root/gtest
