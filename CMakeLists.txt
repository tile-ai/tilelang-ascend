# Copyright(c) Tile-AI Corporation.
# Licensed under the MIT License.
# Learn a lot from the MLC - LLM Project
# https: // github.com/mlc-ai/mlc-llm/blob/main/CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(TILE_LANG C CXX)

# Set default build type to Release if not provided
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Define a custom macro for globbing files with conditional CONFIGURE_DEPENDS
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  macro(tilelang_file_glob glob variable)
    file(${glob} ${variable} CONFIGURE_DEPENDS ${ARGN})
  endmacro()
else()
  macro(tilelang_file_glob glob variable)
    file(${glob} ${variable} ${ARGN})
  endmacro()
endif()

# Handle TVM prebuild path or use default configuration
if(DEFINED TVM_PREBUILD_PATH)
  message(STATUS "TVM_PREBUILD_PATH: ${TVM_PREBUILD_PATH}")

  if(EXISTS ${TVM_PREBUILD_PATH}/config.cmake)
    include(${TVM_PREBUILD_PATH}/config.cmake)
  endif()
else()
  if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
    include(${CMAKE_BINARY_DIR}/config.cmake)
  elseif(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()

  # Set default build type to RelWithDebInfo if not provided
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
    message(STATUS "Setting default build type to ${CMAKE_BUILD_TYPE}")
  endif()
endif()

# include cmake modules
include(CheckCXXCompilerFlag)

# Include code coverage module
include(${CMAKE_SOURCE_DIR}/cmake/modules/CodeCoverage.cmake)

option(USE_ASCEND_SYNC_Z3 "Enable Z3 proofs for Ascend synchronization" ON)
set(Z3_ROOT "" CACHE PATH "Root of the native Z3 installation used by Ascend synchronization")
if(USE_ASCEND_SYNC_Z3)
  include(${CMAKE_SOURCE_DIR}/cmake/modules/FindAscendSyncZ3.cmake)
endif()

# Enable static runtime build if required
if(TILE_LANG_INSTALL_STATIC_LIB)
  set(BUILD_STATIC_RUNTIME ON)
endif()

# Enforce CUDA standard
if(USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
endif()

# Enforce HIP standard
if(USE_ROCM)
  set(CMAKE_HIP_STANDARD 17)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_AMD__ ${CMAKE_CXX_FLAGS}")
endif()

# Enforce C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate TVM prebuild path
if(NOT DEFINED TVM_PREBUILD_PATH)
  if(DEFINED ENV{TVM_PREBUILD_PATH})
    set(TVM_PREBUILD_PATH "$ENV{TVM_PREBUILD_PATH}")
  endif()
endif()

# Locate TVM source directory
if(NOT DEFINED TVM_SOURCE_DIR)
  if(DEFINED ENV{TVM_SOURCE_DIR})
    set(TVM_SOURCE_DIR "$ENV{TVM_SOURCE_DIR}")
  elseif(DEFINED TVM_PREBUILD_PATH)
    set(TVM_SOURCE_DIR "${TVM_PREBUILD_PATH}/..")
  else()
    set(TVM_SOURCE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/tvm)
  endif()
endif()

# Handle TVM prebuild or build TVM from source
if(DEFINED TVM_PREBUILD_PATH)
  message(STATUS "Using prebuilt TVM from ${TVM_PREBUILD_PATH}")
  add_library(tvm SHARED IMPORTED)
  set_target_properties(tvm PROPERTIES
    IMPORTED_LOCATION "${TVM_PREBUILD_PATH}/libtvm.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_PREBUILD_PATH}/../include"
  )
  add_library(tvm_runtime SHARED IMPORTED)
  set_target_properties(tvm_runtime PROPERTIES
    IMPORTED_LOCATION "${TVM_PREBUILD_PATH}/libtvm_runtime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_PREBUILD_PATH}/../include"
  )
else()
  message(STATUS "Building TVM from source at ${TVM_SOURCE_DIR}")
  add_subdirectory(${TVM_SOURCE_DIR} tvm EXCLUDE_FROM_ALL)
endif()

# Collect source files
tilelang_file_glob(GLOB TILE_LANG_SRCS
  src/*.cc
  src/layout/*.cc
  src/transform/*.cc
  src/op/*.cc
  src/target/utils.cc
  src/target/codegen_cpp.cc
  src/target/rt_mod_cpp.cc
  # webgpu doesn't have system dependency
  src/target/codegen_webgpu.cc
)

# The upstream synchronization port lives below the non-recursive transform glob.
list(APPEND TILE_LANG_SRCS
  src/transform/ascend_sync/legacy_adapter.cc
  src/transform/ascend_sync/legacy_composite.cc
  src/transform/ascend_sync/legacy_cross_core.cc
  src/transform/ascend_sync/legacy_pipeline_partition.cc
  src/transform/ascend_sync/legacy_persistent_normalize.cc
  src/transform/ascend_sync/legacy_persistent_pipeline.cc
  src/transform/ascend_sync/legacy_snapshot.cc
  src/transform/ascend_sync/legacy_tail_copy.cc
  src/transform/ascend_sync/physical_storage.cc
  src/transform/ascend_sync/target_policy.cc
  src/transform/ascend_sync/upstream/constraints.cc
  src/transform/ascend_sync/upstream/dependency_analysis.cc
  src/transform/ascend_sync/upstream/insert_sync.cc
  src/transform/ascend_sync/upstream/ir_structure.cc
)

# Keep the optional proof service independent of the top-level transform glob.
list(APPEND TILE_LANG_SRCS src/transform/ascend_sync/z3_proof_service_ffi.cc)
if(USE_ASCEND_SYNC_Z3)
  list(APPEND TILE_LANG_SRCS src/transform/ascend_sync/z3_proof_service.cc)
else()
  list(APPEND TILE_LANG_SRCS src/transform/ascend_sync/z3_proof_service_off.cc)
endif()

# Include CUDA source files if CUDA is enabled
if(USE_CUDA)
  tilelang_file_glob(GLOB TILE_LANG_CUDA_SRCS
    src/runtime/*.cc
    src/target/codegen_cuda.cc
    src/target/rt_mod_cuda.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_CUDA_SRCS})
endif()

if(USE_ASCEND)
  tilelang_file_glob(GLOB TILE_LANG_ASCEND_SRCS
    src/target/codegen_ascend.cc
    src/target/codegen_ascend_pto.cc
    src/target/rt_mod_ascend.cc
    src/target/rt_mod_ascend_pto.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_ASCEND_SRCS})
endif()

# Include ROCm source files if ROCm is enabled
if(USE_ROCM)
  tilelang_file_glob(GLOB TILE_LANG_HIP_SRCS
    src/target/codegen_hip.cc
    src/target/rt_mod_hip.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_HIP_SRCS})
endif()

message(STATUS "Collected source files: ${TILE_LANG_SRCS}")

# Add TileLang object library
add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})

message(STATUS "TVM_SOURCE_DIR: ${TVM_SOURCE_DIR}")
# Include directories for TileLang
set(TILE_LANG_INCLUDES
  ${TVM_SOURCE_DIR}/include
  ${TVM_SOURCE_DIR}/src
  ${TVM_SOURCE_DIR}/3rdparty/dlpack/include
  ${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include
)

# Find CUDA Toolkit
if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)

  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. Please set CUDAToolkit_ROOT.")
  endif()

  message(STATUS "CUDA Toolkit includes: ${CUDAToolkit_INCLUDE_DIRS}")
  set(CUDA_MAJOR_VERSION ${CUDAToolkit_VERSION_MAJOR})
  message(STATUS "Setting CUDA_MAJOR_VERSION=${CUDA_MAJOR_VERSION}")
  add_compile_definitions(CUDA_MAJOR_VERSION=${CUDA_MAJOR_VERSION})
  
  list(APPEND TILE_LANG_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
endif(USE_CUDA)

# Find ROCM Toolkit
if(USE_ROCM)
  find_rocm(${USE_ROCM})
  message(STATUS "USE_ROCM: ${USE_ROCM}")

  if(ROCM_FOUND)
    # always set the includedir
    # avoid global retrigger of cmake
    include_directories(SYSTEM ${ROCM_INCLUDE_DIRS})
    add_definitions(-D__HIP_PLATFORM_HCC__=1)
  else()
    message(FATAL_ERROR "ROCM Toolkit not found. Please set HIP_ROOT.")
  endif(ROCM_FOUND)

  message(STATUS "ROCM Toolkit includes: ${ROCM_INCLUDE_DIRS}")
  list(APPEND TILE_LANG_INCLUDES ${ROCM_INCLUDE_DIRS})
endif(USE_ROCM)

# Define compile-time macros
set(TILE_LANG_COMPILE_DEFS
  DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
  __STDC_FORMAT_MACROS=1
  PICOJSON_USE_INT64
)

# Set target properties for object library
target_include_directories(tilelang_objs PRIVATE ${TILE_LANG_INCLUDES})
target_compile_definitions(tilelang_objs PRIVATE ${TILE_LANG_COMPILE_DEFS})
target_compile_definitions(tilelang_objs PRIVATE -DTILE_LANG_EXPORTS)

# Shared library
add_library(tilelang SHARED $<TARGET_OBJECTS:tilelang_objs>)
target_link_libraries(tilelang PUBLIC tvm_runtime)

# Static library
add_library(tilelang_static STATIC $<TARGET_OBJECTS:tilelang_objs>)
add_dependencies(tilelang_static tvm_runtime)
set_target_properties(tilelang_static PROPERTIES OUTPUT_NAME tilelang)

# Debug build type-specific definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(tilelang PRIVATE "TVM_LOG_DEBUG")
  target_compile_definitions(tilelang_objs PRIVATE "TVM_LOG_DEBUG")
  target_compile_definitions(tilelang_static PRIVATE "TVM_LOG_DEBUG")
endif()

# Module shared library
add_library(tilelang_module SHARED $<TARGET_OBJECTS:tilelang_objs>)
target_link_libraries(tilelang_module PUBLIC tvm)

set(TILELANG_ASCEND_SYNC_Z3_RUNTIME_FILES "")
if(USE_ASCEND_SYNC_Z3)
  # Only the prover implementation includes Z3 headers. Keep the optional
  # dependency out of every other object's compile flags and rebuild scope.
  set_property(SOURCE src/transform/ascend_sync/z3_proof_service.cc
    APPEND PROPERTY INCLUDE_DIRECTORIES "${ASCEND_SYNC_Z3_INCLUDE_DIR}")
  # PUBLIC also propagates the dependency to consumers of the static archive.
  foreach(tilelang_sync_target tilelang tilelang_module tilelang_static)
    target_link_libraries(${tilelang_sync_target} PUBLIC AscendSync::Z3)
  endforeach()
  foreach(tilelang_sync_target tilelang tilelang_module)
    set_property(TARGET ${tilelang_sync_target} APPEND PROPERTY BUILD_RPATH
      "$ORIGIN" "${ASCEND_SYNC_Z3_LIBRARY_DIR}")
    set_property(TARGET ${tilelang_sync_target} APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
    # Wheels need the ELF SONAME as well as the unversioned link name. Some
    # Python Z3 packages ship those as separate regular files, not symlinks.
    foreach(tilelang_sync_z3_name IN LISTS ASCEND_SYNC_Z3_RUNTIME_NAMES)
      add_custom_command(TARGET ${tilelang_sync_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${ASCEND_SYNC_Z3_LIBRARY}"
          "$<TARGET_FILE_DIR:${tilelang_sync_target}>/${tilelang_sync_z3_name}"
        VERBATIM)
    endforeach()
  endforeach()
  foreach(tilelang_sync_z3_name IN LISTS ASCEND_SYNC_Z3_RUNTIME_NAMES)
    string(APPEND TILELANG_ASCEND_SYNC_Z3_RUNTIME_FILES
      "$<TARGET_FILE_DIR:tilelang>/${tilelang_sync_z3_name}\n")
    install(FILES "${ASCEND_SYNC_Z3_LIBRARY}"
      DESTINATION lib${LIB_SUFFIX} RENAME "${tilelang_sync_z3_name}")
  endforeach()
endif()

# setup.py builds targets without running cmake --install. Its packaging
# helper consumes this authoritative manifest, including an empty OFF build.
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/ascend_sync_z3_runtime_files.txt"
  CONTENT "${TILELANG_ASCEND_SYNC_Z3_RUNTIME_FILES}")

install(FILES "${TVM_SOURCE_DIR}/LICENSE" "${TVM_SOURCE_DIR}/NOTICE"
  DESTINATION share/licenses/tilelang/tvm)
install(FILES "${CMAKE_SOURCE_DIR}/src/transform/ascend_sync/NOTICE"
  DESTINATION share/licenses/tilelang/ascend_sync)
install(FILES "${CMAKE_SOURCE_DIR}/src/transform/ascend_sync/licenses/Z3-LICENSE.txt"
  DESTINATION share/licenses/tilelang/z3 RENAME LICENSE.txt)

# Install targets
if(TILE_LANG_INSTALL_STATIC_LIB)
  install(TARGETS tilelang_static tvm_runtime
    LIBRARY DESTINATION lib${LIB_SUFFIX}
  )
else()
  if(DEFINED TVM_PREBUILD_PATH)
    install(TARGETS tilelang tilelang_module
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
  else()
    install(TARGETS tvm tvm_runtime tilelang tilelang_module
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
  endif()
endif()
