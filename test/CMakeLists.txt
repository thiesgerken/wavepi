find_package(GTest)
if (GTEST_FOUND)
    # file(GLOB LIBSOURCES *.cpp)
    SET(LIBSOURCES
            main.cpp
            measurements.cpp
            discretized_function.cpp
            wave_equation.cpp
            wave_equation_adjointness.cpp
            problem_adjointness.cpp
            div_right_hand_side.cpp
        )

    ADD_EXECUTABLE(wavepi_test ${LIBSOURCES})
    TARGET_LINK_LIBRARIES(wavepi_test ${GTEST_BOTH_LIBRARIES} wavepi_forward wavepi_inversion wavepi_base wavepi_norms wavepi_measurements) 
    set_target_properties(wavepi_test PROPERTIES OUTPUT_NAME "wavepi-test")
 
    #include(GoogleTest)
    #if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.10.0)
    #  gtest_discover_tests(wavepi_test)
    #endif()

endif()
