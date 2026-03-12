if(NOT TILELANGIR_MLIR_LIBS)
  return()
endif()

if(DEFINED Python3_EXECUTABLE)
  set(Python3_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Python interpreter for tilelangir pybind11" FORCE)
  message(STATUS "TileLangIR Python (from cache): ${Python3_EXECUTABLE}")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir(), end='')"
  OUTPUT_VARIABLE _pybind11_cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE _pybind11_err
)
if(NOT _pybind11_err AND _pybind11_cmake_dir)
  set(pybind11_DIR "${_pybind11_cmake_dir}" CACHE PATH "pybind11 from selected Python" FORCE)
  message(STATUS "TileLangIR pybind11: ${pybind11_DIR}")
else()
  message(FATAL_ERROR
    "TileLangIR requires pybind11 in the same Python as -DPython3_EXECUTABLE (${Python3_EXECUTABLE}). "
    "Run: ${Python3_EXECUTABLE} -m pip install pybind11==3.0.1")
endif()
find_package(pybind11 CONFIG REQUIRED)

set(_python_mlir_libs -Wl,--start-group ${TILELANGIR_MLIR_LIBS} -Wl,--end-group)

add_tilelangir_library(tilelangir MODULE TileLangIRPasses.cpp)
target_compile_options(tilelangir PRIVATE -frtti)
target_link_libraries(tilelangir PRIVATE
  tilelangir_transforms
  tilelangir_headers
  pybind11::module
  ${_python_mlir_libs}
  ${TILELANGIR_SYSTEM_LIBS}
)
add_dependencies(tilelangir tilelangir_transforms_incgen)

set_target_properties(tilelangir PROPERTIES
  PREFIX ""
  OUTPUT_NAME "libtilelangir"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

install(TARGETS tilelangir
  LIBRARY DESTINATION lib${LIB_SUFFIX}
  RUNTIME DESTINATION bin
)

# Bundle only mlir_core and bishengir from BISHENGIR python_packages/ into build dir. Both are required.
get_filename_component(_BISHENGIR_ROOT_ABS "${BISHENGIR_ROOT_PATH}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
set(_bundle_python_packages "${_BISHENGIR_ROOT_ABS}/python_packages")
if(NOT EXISTS "${_bundle_python_packages}/mlir_core")
  message(FATAL_ERROR "NPUIR build requires python_packages/mlir_core; not found under ${_BISHENGIR_ROOT_ABS}")
endif()
if(NOT EXISTS "${_bundle_python_packages}/bishengir")
  message(FATAL_ERROR "NPUIR build requires python_packages/bishengir; not found under ${_BISHENGIR_ROOT_ABS}")
endif()
add_custom_target(tilelangir_bundle_mlir_python ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${_bundle_python_packages}/mlir_core" "${CMAKE_BINARY_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${_bundle_python_packages}/bishengir" "${CMAKE_BINARY_DIR}"
  COMMENT "Bundling mlir_core and bishengir into ${CMAKE_BINARY_DIR}"
)
add_dependencies(tilelangir_bundle_mlir_python tilelangir)
message(STATUS "TileLangIR: bundling mlir_core and bishengir; PYTHONPATH=${CMAKE_BINARY_DIR} for NPUIR")
