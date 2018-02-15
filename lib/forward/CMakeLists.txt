# file(GLOB LIBSOURCES *.cpp)
SET(LIBSOURCES
        DivRightHandSide.cpp
        DivRightHandSideAdjoint.cpp        
        L2ProductRightHandSide.cpp        
        L2RightHandSide.cpp     
        WaveEquationBase.cpp    
        WaveEquation.cpp       
        WaveEquationAdjoint.cpp
        DistributionRightHandSide.cpp  
        MatrixCreator.cpp
    )
        
add_library(wavepi_forward ${LIBSOURCES})
TARGET_LINK_LIBRARIES(wavepi_forward wavepi_base)

DEAL_II_SETUP_TARGET(wavepi_forward)
