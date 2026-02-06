# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# Generate one .td file: custom_command, install, and append to tablegen_compile_commands.yml.
function(_tilelangir_tablegen_one td_rel tblgen_exe root_abs tblgen_args tblgen_includes current_outputs all_outputs_var)
  if(IS_ABSOLUTE "${td_rel}")
    set(td_abs "${td_rel}")
  else()
    set(td_abs "${CMAKE_CURRENT_SOURCE_DIR}/${td_rel}")
  endif()
  string(REGEX REPLACE "\\.td$" ".h.inc" out_rel "${td_rel}")
  set(out_path "${CMAKE_CURRENT_BINARY_DIR}/${out_rel}")
  get_filename_component(out_dir "${out_path}" DIRECTORY)
  file(MAKE_DIRECTORY "${out_dir}")
  add_custom_command(
    OUTPUT "${out_path}"
    COMMAND ${tblgen_exe} ${tblgen_args} -I "${root_abs}/include"
            "${td_abs}" -o "${out_path}"
    DEPENDS "${td_abs}"
    COMMENT "Building ${out_rel}..."
  )
  list(APPEND current_outputs "${out_path}")
  set(${all_outputs_var} "${current_outputs}" PARENT_SCOPE)
  get_filename_component(out_dir_only "${out_path}" DIRECTORY)
  string(REGEX REPLACE ".*/include/" "include/" install_dest "${out_dir_only}")
  if(install_dest STREQUAL out_dir_only)
    set(install_dest "include")
  endif()
  install(FILES "${out_path}" DESTINATION "${install_dest}" OPTIONAL)
  file(
    APPEND ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml
    "--- !FileInfo:\n"
    "  filepath: \"${td_abs}\"\n"
    "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n")
endfunction()

function(tilelangir_tablegen target_name)
  if(NOT BISHENGIR_ROOT_PATH)
    message(FATAL_ERROR "BISHENGIR_ROOT_PATH required for tilelangir_tablegen")
  endif()
  cmake_parse_arguments(ARG "" "" "TD_FILES;ARGS;EXTRA_INCLUDES;DEPENDS" ${ARGN})
  if(NOT ARG_TD_FILES)
    message(FATAL_ERROR "tilelangir_tablegen: TD_FILES is required (e.g. TD_FILES Passes.td)")
  endif()
  if(NOT ARG_ARGS)
    message(FATAL_ERROR "tilelangir_tablegen: ARGS is required (e.g. ARGS -gen-pass-decls -name TileLangIR)")
  endif()

  get_filename_component(BISHENGIR_ROOT_ABS "${BISHENGIR_ROOT_PATH}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
  set(TBLGEN_EXE "${BISHENGIR_ROOT_ABS}/bin/mlir-tblgen")
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  if(ARG_EXTRA_INCLUDES)
    list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  endif()
  list(REMOVE_ITEM tblgen_includes "")

  set(all_outputs)
  foreach(td_rel ${ARG_TD_FILES})
    _tilelangir_tablegen_one("${td_rel}" "${TBLGEN_EXE}" "${BISHENGIR_ROOT_ABS}" "${ARG_ARGS}" "${tblgen_includes}" "${all_outputs}" all_outputs)
  endforeach()

  add_custom_target(${target_name} DEPENDS ${all_outputs})
  if(ARG_DEPENDS)
    add_dependencies(${target_name} ${ARG_DEPENDS})
  endif()
  list(GET all_outputs -1 last_out)
  set(TILELANGIR_TABLEGEN_OUTPUT "${last_out}" PARENT_SCOPE)
endfunction(tilelangir_tablegen)

# Declare a tilelangir library. Usage:
#   add_tilelangir_library(name sources...)       -> SHARED, appended to TILELANGIR_LIBS (for tools).
#   add_tilelangir_library(name MODULE sources...) -> MODULE (e.g. Python extension), not in TILELANGIR_LIBS.
function(add_tilelangir_library name)
  set(_lib_type SHARED)
  set(_sources ${ARGN})
  if(ARGV1 STREQUAL "MODULE")
    set(_lib_type MODULE)
    list(REMOVE_AT _sources 0)
  endif()
  if(_lib_type STREQUAL "SHARED")
    set_property(GLOBAL APPEND PROPERTY TILELANGIR_LIBS ${name})
  endif()
  add_library(${name} ${_lib_type} ${_sources})
endfunction(add_tilelangir_library)

# Collect all TileLangIR library targets for linking (e.g. in tools).
function(get_tilelangir_link_libraries out_var)
  get_property(_libs GLOBAL PROPERTY TILELANGIR_LIBS)
  set(${out_var} "${_libs}" PARENT_SCOPE)
endfunction(get_tilelangir_link_libraries)

# Declare a TileLangIR tool (executable). Optional: DEPENDS dep1 ... LINK_LIBS lib1 ...
function(add_tilelangir_tool name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    set(sources ${ARG_UNPARSED_ARGUMENTS})
  else()
    set(sources ${name}.cpp)
  endif()
  add_executable(${name} ${sources})
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PRIVATE ${ARG_LINK_LIBS})
  endif()
endfunction(add_tilelangir_tool)

# Declare a TileLangIR tool and add install rule.
function(add_tilelangir_install_tool name)
  add_tilelangir_tool(${name} ${ARGN})
  install(TARGETS ${name}
    RUNTIME DESTINATION bin
  )
endfunction(add_tilelangir_install_tool)
