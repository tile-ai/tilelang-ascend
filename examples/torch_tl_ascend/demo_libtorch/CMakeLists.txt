cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flash_attention_demo LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(Torch_npu REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# TORCH_NPU_SOURCE_DIR must be passed from build.sh
if(NOT DEFINED TORCH_NPU_SOURCE_DIR)
    message(FATAL_ERROR "TORCH_NPU_SOURCE_DIR is not set. Please pass it via cmake: -DTORCH_NPU_SOURCE_DIR=/path/to/torch_npu")
endif()

add_executable(flash_attention_demo flash_attention.cpp)

target_include_directories(flash_attention_demo PRIVATE ${TORCH_NPU_SOURCE_DIR})

target_link_libraries(flash_attention_demo PUBLIC "${TORCH_LIBRARIES}" "${TORCH_NPU_LIBRARIES}")
target_link_libraries(flash_attention_demo PRIVATE dl)  # support dlopen 

set_target_properties(flash_attention_demo PROPERTIES
    CXX_STANDARD 17
    BUILD_RPATH "$ORIGIN;$ORIGIN/.."
    INSTALL_RPATH "$ORIGIN"
)
