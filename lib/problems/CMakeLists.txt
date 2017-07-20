# file(GLOB LIBSOURCES *.cpp)
SET(LIBSOURCES
    L2QProblem.cpp
    WaveProblem.cpp    
    )
    
add_library(wavepi_problems ${LIBSOURCES})
DEAL_II_SETUP_TARGET(wavepi_problems)
TARGET_LINK_LIBRARIES(wavepi_problems wavepi_forward)
