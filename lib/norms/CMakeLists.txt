# file(GLOB LIBSOURCES *.cpp)
SET(LIBSOURCES
    L2L2.cpp
    H1H1.cpp
    H1L2.cpp
    H2L2.cpp
    H2L2PlusL2H1.cpp
    L2Coefficients.cpp
    )
        
add_library(wavepi_norms ${LIBSOURCES})
TARGET_LINK_LIBRARIES(wavepi_norms wavepi_base)

DEAL_II_SETUP_TARGET(wavepi_norms)
