# file(GLOB LIBSOURCES *.cpp)
SET(LIBSOURCES
    PointMeasure.cpp 
    GridPointMeasure.cpp 
    MeasuredValues.cpp
   )
    
add_library(wavepi_measurements ${LIBSOURCES})
DEAL_II_SETUP_TARGET(wavepi_measurements)
TARGET_LINK_LIBRARIES(wavepi_measurements wavepi_base)
