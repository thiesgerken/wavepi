
find_package(GTest)
if (GTEST_FOUND)
    # file(GLOB LIBSOURCES *.cpp)
    SET(LIBSOURCES
            main.cpp
            wave_equation.cpp
            discretized_function.cpp
            adjointness.cpp
        )

    ADD_EXECUTABLE(wavepi_test ${LIBSOURCES})
    TARGET_LINK_LIBRARIES(wavepi_test GTest::GTest GTest::Main wavepi_forward wavepi_inversion wavepi_util) 
    
    include(GoogleTest)
    #gtest_add_tests(target      wavepi_test)
    gtest_discover_tests(wavepi_test)
endif()
