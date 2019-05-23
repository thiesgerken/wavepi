cmake_minimum_required(VERSION 2.8.8)
project(wavePI)
include_directories("include")

option(WAVEPI_WITH_MPI "compile with MPI support" OFF)
option(WAVEPI_WITH_1D "compile with 1D support" OFF)
option(WAVEPI_WITH_2D "compile with 2D support" ON)
option(WAVEPI_WITH_3D "compile with 3D support" ON)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_custom_target(debug
                  COMMAND ${CMAKE_COMMAND}
                          -DCMAKE_BUILD_TYPE=Debug
                          ${CMAKE_SOURCE_DIR}
                  COMMENT "Switch CMAKE_BUILD_TYPE to Debug")

add_custom_target(release
                  COMMAND ${CMAKE_COMMAND}
                          -DCMAKE_BUILD_TYPE=Release
                          ${CMAKE_SOURCE_DIR}
                  COMMENT "Switch CMAKE_BUILD_TYPE to Release")

set(Boost_USE_STATIC_LIBS ON)
set(BOOST_ROOT /usr/local)
find_package(Boost
             1.62
             REQUIRED
             QUIET
             COMPONENTS filesystem program_options regex)

# need to include headers to MPI (deal.II needs them if it was configured that
# way)
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

find_package(deal.II
             9.1.0
             REQUIRED
             QUIET
             HINTS
             ${deal.II_DIR}
             ${DEAL_II_DIR}
             ../
             ../../
             $ENV{DEAL_II_DIR})

deal_ii_initialize_cached_variables()
include_directories(/usr/local/include/deal.II/bundled)

string(REPLACE "-O3"
               ""
               CMAKE_CXX_FLAGS_RELEASE
               "${CMAKE_CXX_FLAGS_RELEASE}")

set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -fext-numeric-literals -O0 -Wall -fPIC")
set(
  CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE} -fext-numeric-literals -O3 -s -Wall -Wno-unused-parameter -fPIC"
  )
set(CMAKE_CXX_STANDARD 14)

string(REPLACE "-O2"
               ""
               CMAKE_CXX_FLAGS_DEBUG
               "${CMAKE_CXX_FLAGS_DEBUG}")
string(REPLACE "-O3"
               ""
               CMAKE_CXX_FLAGS_DEBUG
               "${CMAKE_CXX_FLAGS_DEBUG}")

# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
# set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} -fuse-ld=gold")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Compiling in debug mode")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Compiling in release mode")
else()
  message(STATUS "Compiling in unknown mode!")
endif()

if(WAVEPI_WITH_MPI)
  message(STATUS "Compiling with MPI support")
  add_definitions(-DWAVEPI_MPI)
else()
  message(STATUS "Compiling without MPI support")
endif()

if(WAVEPI_WITH_1D)
  add_definitions(-DWAVEPI_1D)
  message(STATUS "Compiling with 1D support")
endif()

if(WAVEPI_WITH_2D)
  add_definitions(-DWAVEPI_2D)
  message(STATUS "Compiling with 2D support")
endif()

if(WAVEPI_WITH_3D)
  add_definitions(-DWAVEPI_3D)
  message(STATUS "Compiling with 3D support")
endif()

enable_testing()

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(doc)
