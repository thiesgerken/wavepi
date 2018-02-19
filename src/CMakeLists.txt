ADD_EXECUTABLE(wavepi_exe wavepi.cpp)
target_link_libraries(wavepi_exe wavepi wavepi_base wavepi_forward wavepi_inversion Boost::filesystem Boost::program_options)
set_target_properties(wavepi_exe PROPERTIES OUTPUT_NAME "wavepi")

ADD_EXECUTABLE(wavepi_logfilter logfilter.cpp)

ADD_EXECUTABLE(wavepi_demo demo.cpp)
target_link_libraries(wavepi_demo wavepi Boost::filesystem Boost::program_options)
