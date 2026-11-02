# Copyright(c) Tile-AI Corporation.
# Licensed under the MIT License.

# Z3 is optional and must not be discovered or loaded in the OFF build.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(_ascend_sync_z3_root "${Z3_ROOT}")
if(NOT _ascend_sync_z3_root AND DEFINED ENV{Z3_ROOT})
  set(_ascend_sync_z3_root "$ENV{Z3_ROOT}")
endif()
if(NOT _ascend_sync_z3_root)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import pathlib, z3; print(pathlib.Path(z3.__file__).resolve().parent)"
    RESULT_VARIABLE _ascend_sync_z3_python_result
    OUTPUT_VARIABLE _ascend_sync_z3_root
    ERROR_VARIABLE _ascend_sync_z3_python_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _ascend_sync_z3_python_result EQUAL 0)
    message(FATAL_ERROR
      "USE_ASCEND_SYNC_Z3 requires native Z3. Set Z3_ROOT to a directory with "
      "include/z3++.h and lib/libz3.so, or install z3-solver for "
      "${Python3_EXECUTABLE}. ${_ascend_sync_z3_python_error}")
  endif()
endif()
get_filename_component(_ascend_sync_z3_root "${_ascend_sync_z3_root}" ABSOLUTE)

# An explicit root is authoritative: never silently mix its headers with a
# system library, a cached discovery, or another Python environment.
unset(ASCEND_SYNC_Z3_INCLUDE_DIR CACHE)
unset(ASCEND_SYNC_Z3_LIBRARY CACHE)
find_path(ASCEND_SYNC_Z3_INCLUDE_DIR NAMES z3++.h
  PATHS "${_ascend_sync_z3_root}/include" NO_DEFAULT_PATH)
find_file(ASCEND_SYNC_Z3_LIBRARY NAMES libz3.so
  PATHS "${_ascend_sync_z3_root}/lib" "${_ascend_sync_z3_root}/lib64"
  NO_DEFAULT_PATH)
if(NOT ASCEND_SYNC_Z3_INCLUDE_DIR OR NOT ASCEND_SYNC_Z3_LIBRARY)
  message(FATAL_ERROR
    "Invalid Z3 root '${_ascend_sync_z3_root}': expected include/z3++.h and "
    "lib/libz3.so (or lib64/libz3.so). No fallback search was performed.")
endif()

# Python wheels do not necessarily include z3_version.h. Query the selected
# native library through its stable C API, without importing another Z3 module.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import ctypes, sys; lib = ctypes.CDLL(sys.argv[1]); version = [ctypes.c_uint() for _ in range(4)]; lib.Z3_get_version.argtypes = [ctypes.POINTER(ctypes.c_uint)] * 4; lib.Z3_get_version.restype = None; lib.Z3_get_version(*(ctypes.byref(v) for v in version)); print('.'.join(str(v.value) for v in version))"
    "${ASCEND_SYNC_Z3_LIBRARY}"
  RESULT_VARIABLE _ascend_sync_z3_version_result
  OUTPUT_VARIABLE ASCEND_SYNC_Z3_VERSION
  ERROR_VARIABLE _ascend_sync_z3_version_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _ascend_sync_z3_version_result EQUAL 0 OR
   NOT ASCEND_SYNC_Z3_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR
    "Cannot query native Z3 version at '${ASCEND_SYNC_Z3_LIBRARY}': "
    "${_ascend_sync_z3_version_error}")
endif()
if(ASCEND_SYNC_Z3_VERSION VERSION_LESS "4.13.0" OR
   NOT ASCEND_SYNC_Z3_VERSION VERSION_LESS "4.15.5")
  message(FATAL_ERROR
    "Ascend synchronization requires Z3 >= 4.13.0 and < 4.15.5; found "
    "${ASCEND_SYNC_Z3_VERSION} at '${ASCEND_SYNC_Z3_LIBRARY}'.")
endif()

get_filename_component(ASCEND_SYNC_Z3_LIBRARY_DIR "${ASCEND_SYNC_Z3_LIBRARY}" DIRECTORY)
get_filename_component(_ascend_sync_z3_link_name "${ASCEND_SYNC_Z3_LIBRARY}" NAME)
set(ASCEND_SYNC_Z3_RUNTIME_NAMES "${_ascend_sync_z3_link_name}")
if(NOT CMAKE_READELF)
  find_program(CMAKE_READELF NAMES readelf llvm-readelf)
endif()
if(NOT CMAKE_READELF)
  message(FATAL_ERROR "readelf is required to package the selected Z3 library's ELF SONAME.")
endif()
execute_process(COMMAND "${CMAKE_READELF}" -d "${ASCEND_SYNC_Z3_LIBRARY}"
  RESULT_VARIABLE _ascend_sync_z3_readelf_result
  OUTPUT_VARIABLE _ascend_sync_z3_dynamic
  ERROR_VARIABLE _ascend_sync_z3_readelf_error)
if(NOT _ascend_sync_z3_readelf_result EQUAL 0)
  message(FATAL_ERROR "Cannot inspect Z3 SONAME: ${_ascend_sync_z3_readelf_error}")
endif()
if(_ascend_sync_z3_dynamic MATCHES "\\(SONAME\\)[^\n]*\\[([^]]+)\\]")
  set(ASCEND_SYNC_Z3_SONAME "${CMAKE_MATCH_1}")
  get_filename_component(_ascend_sync_z3_soname_basename "${ASCEND_SYNC_Z3_SONAME}" NAME)
  if(NOT _ascend_sync_z3_soname_basename STREQUAL ASCEND_SYNC_Z3_SONAME)
    message(FATAL_ERROR "The selected Z3 SONAME must be a basename, not an absolute dependency.")
  endif()
  list(APPEND ASCEND_SYNC_Z3_RUNTIME_NAMES "${ASCEND_SYNC_Z3_SONAME}")
else()
  message(FATAL_ERROR
    "The selected Z3 library has no ELF SONAME; a relocatable runtime is required.")
endif()
list(REMOVE_DUPLICATES ASCEND_SYNC_Z3_RUNTIME_NAMES)

add_library(AscendSync::Z3 SHARED IMPORTED GLOBAL)
set_target_properties(AscendSync::Z3 PROPERTIES
  IMPORTED_LOCATION "${ASCEND_SYNC_Z3_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_SYNC_Z3_INCLUDE_DIR}")
if(ASCEND_SYNC_Z3_SONAME)
  set_target_properties(AscendSync::Z3 PROPERTIES IMPORTED_SONAME "${ASCEND_SYNC_Z3_SONAME}")
endif()
message(STATUS "Ascend sync Z3 ${ASCEND_SYNC_Z3_VERSION}: ${ASCEND_SYNC_Z3_LIBRARY}")
