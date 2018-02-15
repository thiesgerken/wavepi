cmake_minimum_required(VERSION 2.8.8)
project(wavePI)
include_directories("include")

set(BUILD_SHARED_LIBS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_custom_target(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

add_custom_target(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

find_package(deal.II 8.5.0 REQUIRED QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

find_package(Boost 1.56 REQUIRED COMPONENTS
             filesystem program_options)

DEAL_II_INITIALIZE_CACHED_VARIABLES()

string (REPLACE "-O3" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -s -Wall -Wno-unused-parameter")
set(CMAKE_CXX_STANDARD 14)

string (REPLACE "-O2" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
string (REPLACE "-O3" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})

set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} -fuse-ld=gold")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")

if(WAVEPI_WITH_MPI)
    message(STATUS "Compiling with MPI support")
    add_definitions(-DWAVEPI_MPI)
endif()

enable_testing()

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(doc)
