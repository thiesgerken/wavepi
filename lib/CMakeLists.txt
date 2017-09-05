add_subdirectory(util)
add_subdirectory(forward)
add_subdirectory(inversion)
# add_subdirectory(problems)
add_subdirectory(measurements)

SET(LIBSOURCES
    WavePI.cpp 
    SettingsManager.cpp
    )
    
add_library(wavepi ${LIBSOURCES})
DEAL_II_SETUP_TARGET(wavepi)
TARGET_LINK_LIBRARIES(wavepi wavepi_measurements wavepi_forward wavepi_inversion wavepi_util)

