ADD_EXECUTABLE(wavepi wavepi.cpp)
DEAL_II_SETUP_TARGET(wavepi)
TARGET_LINK_LIBRARIES(wavepi wavepi_forward wavepi_problems)

ADD_EXECUTABLE(logfilter logfilter.cpp)