
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
    
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.10)
      gtest_discover_tests(wavepi_test)
    endif()
endif()
