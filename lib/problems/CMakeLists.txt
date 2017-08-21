# file(GLOB LIBSOURCES *.cpp)
SET(LIBSOURCES
    L2QProblem.cpp
    L2CProblem.cpp
    L2AProblem.cpp
    L2NuProblem.cpp  
    )
    
add_library(wavepi_problems ${LIBSOURCES})
DEAL_II_SETUP_TARGET(wavepi_problems)
TARGET_LINK_LIBRARIES(wavepi_problems wavepi_forward wavepi_inversion)
