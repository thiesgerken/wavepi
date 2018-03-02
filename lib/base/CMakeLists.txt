# file(GLOB LIBSOURCES *.cpp)
SET(LIBSOURCES
        ConstantMesh.cpp          
        SpaceTimeMesh.cpp
        AdaptiveMesh.cpp
        DiscretizedFunction.cpp 
        MacroFunctionParser.cpp 
        RadialParsedFunction.cpp 
        Transformation.cpp
        Util.cpp   
        Norm.cpp
    )
        
add_library(wavepi_base ${LIBSOURCES})

DEAL_II_SETUP_TARGET(wavepi_base)
