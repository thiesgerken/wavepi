# file(GLOB LIBSOURCES *.cpp)
SET(LIBSOURCES
        ToleranceChoice.cpp
        ConstantToleranceChoice.cpp
        RiederToleranceChoice.cpp
   )
    
add_library(wavepi_inversion ${LIBSOURCES})
# TARGET_LINK_LIBRARIES(wavepi_inversion wavepi_forward)

DEAL_II_SETUP_TARGET(wavepi_inversion)
