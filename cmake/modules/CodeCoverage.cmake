# CodeCoverage.cmake - CMake module for code coverage support

# Only enable coverage if explicitly requested
if(NOT ENABLE_COVERAGE)
    message(STATUS "Coverage not enabled (ENABLE_COVERAGE not set)")
    return()
endif()

message(STATUS "Enabling code coverage support")

# Check for coverage tools
find_program(GCOV_PATH gcov)
find_program(LCOV_PATH lcov)
find_program(GENHTML_PATH genhtml)

if(NOT GCOV_PATH)
    message(WARNING "gcov not found, coverage will not work properly")
endif()

if(NOT LCOV_PATH)
    message(WARNING "lcov not found, coverage will not work properly")
endif()

# Coverage compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")

# Function to add coverage target
function(add_coverage_target target_name)
    add_custom_target(${target_name}_coverage
        COMMAND ${LCOV_PATH} --capture --directory ${CMAKE_BINARY_DIR} --output-file ${target_name}_coverage.info
        COMMAND ${LCOV_PATH} --extract ${target_name}_coverage.info "*/src/*" --output-file ${target_name}_coverage.info
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating coverage data for ${target_name}"
    )
    
    add_dependencies(${target_name}_coverage ${target_name})
endfunction()
