CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

# If deal.II can not be found: either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake
# or set an environment variable "DEAL_II_DIR" that contains this path.
FIND_PACKAGE(deal.II 8.5.0 REQUIRED QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# cmake -G "Eclipse CDT4 - Unix Makefiles" -D_ECLIPSE_VERSION=4.7 ~/Promotion/WavePI/
# set(CMAKE_ECLIPSE_VERSION "4.7.0")

# the commit's SHA1, and whether the building workspace was dirty or not
execute_process(COMMAND
  git describe --match=NeVeRmAtCh --always --abbrev=40 --dirty
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_SHA1
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# the date of the commit
execute_process(COMMAND
  git log -1 --format=%ad --date=local
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_DATE
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# the subject of the commit
execute_process(COMMAND
  git log -1 --format=%s
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_COMMIT_SUBJECT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

PROJECT(wavePI)
include_directories("include")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fextended-identifiers -DGIT_DATE='${GIT_DATE}' -DGIT_SHA1='${GIT_SHA1}'")

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(test)
