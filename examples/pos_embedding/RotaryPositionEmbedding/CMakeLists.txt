# CMakeLists.txt — Build perf_rope_ascendc (C++ aclnn driver for msprof).
#
# Auto-detects CANN libraries (libascendcl + libopapi) from ASCEND_HOME_PATH.
# No manual -L/-l needed.
#
# Usage:
#   mkdir build && cd build && cmake .. && make -j
#   # or via wrapper: bash build_perf_rope.sh

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(perf_rope_ascendc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Locate CANN ---
# Priority: ASCEND_HOME_PATH > ASCEND_TOOLKIT_HOME > /usr/local/Ascend/ascend-toolkit/latest
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(CANN_HOME "$ENV{ASCEND_HOME_PATH}")
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(CANN_HOME "$ENV{ASCEND_TOOLKIT_HOME}")
else()
    set(CANN_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()

message(STATUS "CANN_HOME: ${CANN_HOME}")

if(NOT IS_DIRECTORY "${CANN_HOME}")
    message(FATAL_ERROR "CANN not found at ${CANN_HOME}. "
        "Set ASCEND_HOME_PATH (run 'source set_env.sh') before cmake.")
endif()

# CANN include: $CANN_HOME/include
set(CANN_INCLUDE "${CANN_HOME}/include")
if(NOT EXISTS "${CANN_INCLUDE}/acl/acl.h")
    message(FATAL_ERROR "acl/acl.h not found at ${CANN_INCLUDE}. Check CANN install.")
endif()

# Verify aclnn header for rotary_position_embedding exists
if(NOT EXISTS "${CANN_INCLUDE}/aclnnop/aclnn_rotary_position_embedding.h")
    message(FATAL_ERROR
        "aclnnop/aclnn_rotary_position_embedding.h not found.\n"
        "Install the rotary_position_embedding op package first:\n"
        "  cd <ops-transformer>\n"
        "  bash build.sh --pkg --soc=ascend910b --ops=rotary_position_embedding --vendor_name=custom -j16\n"
        "  ./build_out/*.run --install-path=${CANN_HOME}")
endif()

# CANN lib64 layout varies by CANN version:
#   - $CANN_HOME/lib64                      (older / flat)
#   - $CANN_HOME/<arch>-linux/lib64         (newer, e.g. aarch64-linux, x86_64-linux)
# find_library searches all of them.
set(CANN_LIB_SEARCH_PATHS
    "${CANN_HOME}/lib64"
    "${CANN_HOME}/aarch64-linux/lib64"
    "${CANN_HOME}/x86_64-linux/lib64"
    "${CANN_HOME}/arm64-linux/lib64")

find_library(ASCENDCL_LIB
    NAMES ascendcl
    PATHS ${CANN_LIB_SEARCH_PATHS}
    NO_DEFAULT_PATH)
if(NOT ASCENDCL_LIB)
    message(FATAL_ERROR "libascendcl.so not found. Searched:\n${CANN_LIB_SEARCH_PATHS}")
endif()

find_library(OPAPI_LIB
    NAMES opapi
    PATHS ${CANN_LIB_SEARCH_PATHS}
    NO_DEFAULT_PATH)
if(NOT OPAPI_LIB)
    message(FATAL_ERROR "libopapi.so not found. Searched:\n${CANN_LIB_SEARCH_PATHS}")
endif()

# libnnopbase.so: provides aclCreateTensor / aclDestroyTensor (aclTensor API)
find_library(NNOPBASE_LIB
    NAMES nnopbase
    PATHS ${CANN_LIB_SEARCH_PATHS}
    NO_DEFAULT_PATH)
if(NOT NNOPBASE_LIB)
    message(FATAL_ERROR "libnnopbase.so not found. Searched:\n${CANN_LIB_SEARCH_PATHS}")
endif()

# Get the directory of the found lib for rpath
get_filename_component(CANN_LIB_DIR "${ASCENDCL_LIB}" DIRECTORY)
message(STATUS "libascendcl: ${ASCENDCL_LIB}")
message(STATUS "libopapi:    ${OPAPI_LIB}")
message(STATUS "libnnopbase: ${NNOPBASE_LIB}")
message(STATUS "lib dir:     ${CANN_LIB_DIR}")

# --- Build the executable ---
add_executable(perf_rope_ascendc perf_rope_ascendc.cpp)

target_include_directories(perf_rope_ascendc PRIVATE "${CANN_INCLUDE}")

target_link_libraries(perf_rope_ascendc PRIVATE
    ${OPAPI_LIB}
    ${NNOPBASE_LIB}
    ${ASCENDCL_LIB}
)

# Embed rpath so the binary finds CANN libs at runtime without LD_LIBRARY_PATH
set_target_properties(perf_rope_ascendc PROPERTIES
    BUILD_RPATH "${CANN_LIB_DIR}"
    INSTALL_RPATH "${CANN_LIB_DIR}"
)
