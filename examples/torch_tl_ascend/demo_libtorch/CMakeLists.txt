cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flash_attention_demo LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(Torch_npu REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(flash_attention_demo flash_attention.cpp)
add_library(op SHARED IMPORTED)

set_target_properties(op PROPERTIES IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/libop.so")

target_link_libraries(flash_attention_demo PUBLIC "${TORCH_LIBRARIES}" "${TORCH_NPU_LIBRARIES}")
target_link_libraries(flash_attention_demo PRIVATE op)

set_property(TARGET flash_attention_demo PROPERTY CXX_STANDARD 17)
set_property(TARGET flash_attention_demo PROPERTY BUILD_RPATH "$ORIGIN")
set_property(TARGET flash_attention_demo PROPERTY INSTALL_RPATH "$ORIGIN")