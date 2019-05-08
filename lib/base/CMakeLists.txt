# file(GLOB LIBSOURCES *.cpp)
set(LIBSOURCES
    ConstantMesh.cpp
    FunctionParser.cpp
    SpaceTimeMesh.cpp
    AdaptiveMesh.cpp
    DiscretizedFunction.cpp
    MacroFunctionParser.cpp
    RadialParsedFunction.cpp
    Transformation.cpp
    Util.cpp)

add_library(wavepi_base STATIC ${LIBSOURCES})
deal_ii_setup_target(wavepi_base)
