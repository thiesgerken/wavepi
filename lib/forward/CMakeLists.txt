# file(GLOB LIBSOURCES *.cpp)
SET(LIBSOURCES
        DivRightHandSide.cpp
        RightHandSide.cpp
        ConstantMesh.cpp               
        L2ProductRightHandSide.cpp  
        SpaceTimeMesh.cpp
        DiscretizedFunction.cpp        
        L2RightHandSide.cpp         
        WaveEquation.cpp
        DistributionRightHandSide.cpp  
        MatrixCreator.cpp
    )
        
add_library(wavepi_forward ${LIBSOURCES})
DEAL_II_SETUP_TARGET(wavepi_forward)
