# file(GLOB LIBSOURCES *.cpp)
set(LIBSOURCES
    ToleranceChoice.cpp
    ConstantToleranceChoice.cpp
    RiederToleranceChoice.cpp
    WinklerToleranceChoice.cpp
    MaxIterChoice.cpp
    ConstantMaxIterChoice.cpp
    FibonacciMaxIterChoice.cpp)

add_library(wavepi_inversion STATIC ${LIBSOURCES})
deal_ii_setup_target(wavepi_inversion)
