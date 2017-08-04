ADD_EXECUTABLE(wavepi wavepi.cpp)
DEAL_II_SETUP_TARGET(wavepi)
target_link_libraries(wavepi wavepi_forward wavepi_inversion wavepi_problems boost_filesystem)

ADD_EXECUTABLE(logfilter logfilter.cpp)