# file(GLOB LIBSOURCES *.cpp)
SET(LIBSOURCES
    ConvolutionMeasure.cpp 
    DeltaMeasure.cpp
    SensorValues.cpp
    SensorDistribution.cpp
    GridDistribution.cpp
    CubeBoundaryDistribution.cpp
   )
    
add_library(wavepi_measurements ${LIBSOURCES})
DEAL_II_SETUP_TARGET(wavepi_measurements)
TARGET_LINK_LIBRARIES(wavepi_measurements wavepi_base)
