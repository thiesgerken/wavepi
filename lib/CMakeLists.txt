add_subdirectory(util)
add_subdirectory(forward)
add_subdirectory(inversion)
add_subdirectory(problems)

SET(LIBSOURCES
    WavePI.cpp 
    )
    
add_library(wavepi ${LIBSOURCES})
DEAL_II_SETUP_TARGET(wavepi)
TARGET_LINK_LIBRARIES(wavepi wavepi_forward wavepi_problems wavepi_inversion wavepi_util)

