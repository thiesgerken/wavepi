# WavePI (Parameter Identification for Wave Equations)

[![build status](https://git.thiesgerken.de/thies/wavepi/badges/master/build.svg)](https://git.thiesgerken.de/thies/wavepi/commits/master)

© 2017 Thies Gerken (@thies), University of Bremen, `tgerken@math.uni-bremen.de`

Developed as part of my PhD-Project [*Dynamic Inverse Problems for Wave Equations*](https://git.thiesgerken.de/thies/promotion)

This is a work in progress, cf. [issue tracker](https://git.thiesgerken.de/thies/wavepi/issues).

## Dependencies

 * `cmake   >= 2.8.8`
 * `deal-ii >= 8.5.0` 
 * `boost   >= 1.56 `
 * `gtest   >= 1.8.0` (optional)

Note that deali-ii has to be configured with [TBB](https://www.threadingbuildingblocks.org/) support (either bundled or external)

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

Change Build type to release (no assertions, typically runs 10 times faster):

```shell
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Use the same command with `Debug` to go back. There are also `make` targets that switch the build type. 

## Tests

This project uses [Google Test](https://github.com/google/googletest). Run the test suite using the binary `wavepi_test` (only built if `gtest` was found). You can also list all tests (`--gtest_list_tests`) and only run a subset of them (`--gtest_filter="[filter]"`, wildcards are allowed). Currently, a few of the tests should fail (L2 Adjoint to the wave equation by integrating backwards is not as good as `WaveEquationAdjoint`, and is not even correct if $`\nu\neq 0`$).

## Remarks on the Code

It is common C++ practice to put all the code of templated classes into the header file, because the compiler needs to instantiate them for every compilation unit. For classes/functions that only depend on the dimensions I ignored this and just added instances for one, two and three dimensions to increase compilation speed.  

## Code size

Use `cloc` to count the lines of code in this project. To obtain meaningful results, exclude the build directory (assumed to be in `build`) and `doc`:

```shell
cloc . --exclude-dir=build,doc
``` 

## Shell Autocompletion (ZSH)

Put (or symlink) [completions.zsh](completions.zsh) in `~/.zsh-completions` and make sure you have the following lines in `~/.zshrc`:

```shell
# folder of all of your autocomplete functions
fpath=($HOME/.zsh-completions $fpath)

# enable autocomplete function
autoload -U compinit
compinit
``` 
