# file(GLOB LIBSOURCES *.cpp)
set(LIBSOURCES
    DivRightHandSide.cpp
    DivRightHandSideAdjoint.cpp
    L2ProductRightHandSide.cpp
    L2RightHandSide.cpp
    VectorRightHandSide.cpp
    AbstractEquation.cpp
    AbstractEquationAdjoint.cpp
    WaveEquationBase.cpp
    WaveEquation.cpp
    WaveEquationAdjoint.cpp
    DistributionRightHandSide.cpp
    MatrixCreator.cpp)

add_library(wavepi_forward ${LIBSOURCES})
target_link_libraries(wavepi_forward wavepi_base)
deal_ii_setup_target(wavepi_forward)
