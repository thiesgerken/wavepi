file(GLOB LIBSOURCES *.cpp)

add_library(wavepi_forward ${LIBSOURCES})
DEAL_II_SETUP_TARGET(wavepi_forward)
