# file(GLOB LIBSOURCES *.cpp)
SET(LIBSOURCES WavePI.cpp MatrixCreator.cpp GridTools.cpp)
        
find_program(GIT_SCM git DOC "Git version control")
mark_as_advanced(GIT_SCM)
find_file(GITDIR NAMES .git PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
if (GIT_SCM AND GITDIR)
    message(STATUS "Updating git commit information")

    # the commit's SHA1, and whether the building workspace was dirty or not
    execute_process(COMMAND
      git describe --match=NeVeRmAtCh --always --abbrev=40 --dirty
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE GIT_COMMIT_DESCRIPTION
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    # the date of the commit
    execute_process(COMMAND
      git log -1 --format=%ad --date=local
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE GIT_COMMIT_DATE
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    # the subject of the commit
    execute_process(COMMAND
      git log -1 --format=%s
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE GIT_COMMIT_SUBJECT
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    # the current working branch
    execute_process(
      COMMAND git rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE GIT_BRANCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # dirty/untracked files
    execute_process(
      COMMAND git status -s
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE GIT_STATUS
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
     
    string(REGEX REPLACE "\n" "\\\\n" GIT_STATUS "${GIT_STATUS}")
else()
    # No version control
    # e.g. when the software is built from a source tarball
    message(STATUS "Not under version control, Version.cpp will not have git information in it.")
endif()

string(TIMESTAMP BUILD_DATE UTC)
  
configure_file(
  ${CMAKE_SOURCE_DIR}/lib/util/Version.cpp.in
  ${CMAKE_BINARY_DIR}/lib/util/Version.cpp
)

add_custom_target(touch_version_cpp
    COMMAND touch ${CMAKE_SOURCE_DIR}/lib/util/Version.cpp.in
)

list(APPEND LIBSOURCES "${CMAKE_CURRENT_BINARY_DIR}/Version.cpp")
    
add_library(wavepi_util ${LIBSOURCES})
add_dependencies(wavepi_util touch_version_cpp)
DEAL_II_SETUP_TARGET(wavepi_util)
target_link_libraries(wavepi_util boost_filesystem)
