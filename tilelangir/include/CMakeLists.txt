add_subdirectory(tilelangir)

get_filename_component(_tilelangir_gen_abs "${TILELANGIR_TRANSFORMS_GEN_INCLUDE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_BINARY_DIR}")
get_filename_component(_bisherir_include_abs "${BISHENGIR_INCLUDE_DIRS}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
add_library(tilelangir_headers INTERFACE)
target_include_directories(tilelangir_headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${_tilelangir_gen_abs}>
    $<BUILD_INTERFACE:${_bisherir_include_abs}>
)

