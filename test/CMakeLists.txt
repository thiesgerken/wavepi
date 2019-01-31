set(LIBSOURCES
    main.cpp
    measurements.cpp
    discretized_function.cpp
    wave_equation.cpp
    wave_equation_adjointness.cpp
    problem_adjointness.cpp
    div_right_hand_side.cpp)

add_executable(wavepi_test ${LIBSOURCES})
target_link_libraries(wavepi_test
                      gtest
                      gtest_main
                      pthread
                      wavepi_forward
                      wavepi_inversion
                      wavepi_base
                      wavepi_norms
                      wavepi_measurements)
set_target_properties(wavepi_test PROPERTIES OUTPUT_NAME "wavepi-test")
