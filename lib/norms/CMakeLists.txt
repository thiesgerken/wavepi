# file(GLOB LIBSOURCES *.cpp)
set(LIBSOURCES
    L2L2.cpp
    H1H1.cpp
    H1L2.cpp
    H2L2.cpp
    H2L2PlusL2H1.cpp
    L2Coefficients.cpp
    LPWrapper.cpp)

add_library(wavepi_norms ${LIBSOURCES})
target_link_libraries(wavepi_norms wavepi_base)

deal_ii_setup_target(wavepi_norms)
