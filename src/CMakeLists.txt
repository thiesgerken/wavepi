add_executable(wavepi_exe wavepi.cpp)
target_link_libraries(wavepi_exe
                      wavepi
                      wavepi_base
                      wavepi_forward
                      wavepi_inversion
                      ${Boost_LIBRARIES})
set_target_properties(wavepi_exe PROPERTIES OUTPUT_NAME "wavepi")

add_executable(wavepi_logfilter logfilter.cpp)
set_target_properties(wavepi_logfilter
                      PROPERTIES OUTPUT_NAME "wavepi-logfilter")

add_executable(wavepi_demo demo.cpp)
target_link_libraries(wavepi_demo wavepi ${Boost_LIBRARIES})
set_target_properties(wavepi_demo PROPERTIES OUTPUT_NAME "wavepi-demo")
