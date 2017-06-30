ADD_EXECUTABLE(wave_test test.cpp)
DEAL_II_SETUP_TARGET(wave_test)
TARGET_LINK_LIBRARIES(wave_test wavepi)
