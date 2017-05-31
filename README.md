# WavePI (Parameter Identification for Wave Equations)

![](https://git.thiesgerken.de/thies/wavepi/badges/master/build.svg)

(C) 2017 Thies Gerken (@thies), University of Bremen, `tgerken@math.uni-bremen.de`

Developed as part of my PhD-Project [*Dynamic Inverse Problems for Wave Equations*](https://git.thiesgerken.de/thies/promotion)

This is a work in progress, cf. [Todo.md](Todo.md).

## Dependencies

 * `cmake   >= 2.8.8`
 * `deal-ii == 8.5.0`

## How to Build

Compile using `N` parallel jobs:

```shell
mkdir build
cd build
cmake ..
make -jN
```

Generate Eclipse Project Files: (Don't do this in a child directory)

```shell
cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=4.6 /path/to/wavepi
```
