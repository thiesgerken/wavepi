# WavePI (Parameter Identification for Wave Equations)

![](https://git.thiesgerken.de/thies/wavepi/badges/master/build.svg)

(C) 2017 Thies Gerken (@thies), University of Bremen, `tgerken@math.uni-bremen.de`

Developed as part of my PhD-Project [*Dynamic Inverse Problems for Wave Equations*](https://git.thiesgerken.de/thies/promotion)

This is a work in progress, cf. [Todo.md](Todo.md).

## Dependencies

 * `cmake   >= 2.8.8`
 * `deal-ii >= 8.5.0`

## How to Build

Compile using `N` parallel jobs:

```shell
mkdir build
cd build
cmake ..
make -jN
```

Generate Eclipse Project Files: (Do not do this in a child directory)

```shell
cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=4.7 /path/to/wavepi
```

## Remarks on the Code

It is common C++ practice to put all the code of templated classes into the header file, because the compiler needs to instantiate them for every compilation unit. For classes/functions that only depend on the dimensions I ignored this and just added instances for one, two and three dimensions to increase compilation speed.  
