# file(GLOB LIBSOURCES *.cpp)
set(LIBSOURCES
    ConvolutionMeasure.cpp
    DeltaMeasure.cpp
    SensorValues.cpp
    SensorDistribution.cpp
    GridDistribution.cpp
    CubeBoundaryDistribution.cpp)

add_library(wavepi_measurements STATIC ${LIBSOURCES})
target_link_libraries(wavepi_measurements wavepi_base)
deal_ii_setup_target(wavepi_measurements)
