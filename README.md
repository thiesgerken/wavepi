# WavePI (Parameter Identification for Wave Equations)

(C) 2017 Thies Gerken (@thies), University of Bremen, `tgerken@math.uni-bremen.de`

Developed as part of my PhD-Project [*Dynamic Inverse Problems for Wave Equations*](https://git.thiesgerken.de/thies/promotion)

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

Generate Eclipse Project Files:

```shell
cmake -G "Eclipse CDT4 - Unix Makefiles" /path/to/wavepi            
```