# file(GLOB LIBSOURCES *.cpp)
SET(LIBSOURCES
        DivRightHandSide.cpp
        DivRightHandSideAdjoint.cpp
        ConstantMesh.cpp               
        L2ProductRightHandSide.cpp  
        SpaceTimeMesh.cpp
        AdaptiveMesh.cpp
        DiscretizedFunction.cpp        
        L2RightHandSide.cpp     
        WaveEquationBase.cpp    
        WaveEquation.cpp       
        WaveEquationAdjoint.cpp
        DistributionRightHandSide.cpp  
        Measure.cpp
    )
        
add_library(wavepi_forward ${LIBSOURCES})
DEAL_II_SETUP_TARGET(wavepi_forward)
