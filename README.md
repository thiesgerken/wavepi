# WavePI (Parameter Identification for Wave Equations)

© 2017-2019 Thies Gerken, University of Bremen, `tgerken@math.uni-bremen.de`

Developed as part of my PhD-Project [_Dynamic Inverse Problems for Wave Phenomena_](https://nbn-resolving.de/urn:nbn:de:gbv:46-00107730-18)

![sample reconstruction](reconstruction.png)

## Dependencies

- `cmake >= 2.8.8`
- `deal.II >= 9.1.0-pre`
- `boost >= 1.62`
- `gtest >= 1.8.0` (optional)

Note that `deal.II` has to be configured with [TBB](https://www.threadingbuildingblocks.org/), MPI and UMFPACK support (either bundled or external). I use

```shell
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DDEAL_II_WITH_MPI=ON
```

for configuring `deal.II`.

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
cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=4.7 -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j1 /path/to/wavepi
```

Change Build type to release (no assertions, typically runs 10 times faster):

```shell
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Use the same command with `Debug` to go back. There are also `make` targets that switch the build type.
To only build the documentation (Doxygen), run `make doc` inside the build directory. The command `make run-doc` will also open the result in a Browser

## MPI

Add `-DWAVEPI_WITH_MPI` to the `cmake` invokation to enable MPI support. In this case, MPI is used to parallelize PDE solutions for different right hand sides.

## Tests

This project uses [Google Test](https://github.com/google/googletest). Run the test suite using the binary `wavepi_test` (only built if `gtest` was found). You can also list all tests (`--gtest_list_tests`) and only run a subset of them (`--gtest_filter="[filter]"`, wildcards are allowed). Currently, a few of the tests should fail (L2 Adjoint to the wave equation by integrating backwards is not as good as `WaveEquationAdjoint`, and is not even correct if $`\nu\neq 0`$).

When using `CMake >= 3.10`, one can also run the tests using [`ctest`](https://cmake.org/cmake/help/latest/manual/ctest.1.html). Just run `ctest` in the build directory. `ctest -V` also shows test output, `ctest -N` lists all tests and `ctest -R <regex>` runs all tests that match the specified regex (use `.*` instead of `*`!). If you want colors using `ctest`, run `export GTEST_COLOR=1` beforehand.

## Remarks on the Code

It is common C++ practice to put all the code of templated classes into the header file, because the compiler needs to instantiate them for every compilation unit. For classes/functions that only depend on the space dimension as a template parameter, I ignored this rule and just added instances for one, two and three dimensions to increase compilation speed, as is common also in `deal.II`.

## Shell Autocompletion (ZSH)

Put (or symlink) [completions.zsh](completions.zsh) in `~/.zsh-completions` and make sure you have the following lines in `~/.zshrc`:

```shell
fpath=($HOME/.zsh-completions $fpath)
autoload -U compinit
compinit
```
