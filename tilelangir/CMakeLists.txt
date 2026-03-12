cmake_minimum_required(VERSION 3.18)

project(TileLangIR LANGUAGES CXX)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddTileLangIR)

# MLIR/BiShengIR libs for linking
set(TILELANGIR_MLIR_LIBS "")
set(TILELANGIR_MLIR_LIBS_SHARED "")
if(BISHENGIR_LIBRARY_DIRS)
  foreach(_dir ${BISHENGIR_LIBRARY_DIRS})
    get_filename_component(_dir_abs "${_dir}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    file(GLOB _libs "${_dir_abs}/*.a")
    list(APPEND TILELANGIR_MLIR_LIBS ${_libs})
    file(GLOB _libs_so "${_dir_abs}/*.so")
    list(APPEND TILELANGIR_MLIR_LIBS_SHARED ${_libs_so})
  endforeach()
elseif(BISHENGIR_ROOT_PATH)
  get_filename_component(_lib_dir "${BISHENGIR_ROOT_PATH}/lib" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
  file(GLOB TILELANGIR_MLIR_LIBS "${_lib_dir}/*.a")
  file(GLOB TILELANGIR_MLIR_LIBS_SHARED "${_lib_dir}/*.so")
endif()
set(TILELANGIR_SYSTEM_LIBS z zstd dl m rt)

# Add subdirectories
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(python)
