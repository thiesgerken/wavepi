
find_package(GTest)
if (GTEST_FOUND)
       file(GLOB LIBSOURCES *.cpp)

        ADD_EXECUTABLE(wavepi_test ${LIBSOURCES})
        DEAL_II_SETUP_TARGET(wavepi_test)
        TARGET_LINK_LIBRARIES(wavepi_test ${GTEST_BOTH_LIBRARIES} wavepi_forward wavepi_inversion)
endif()
