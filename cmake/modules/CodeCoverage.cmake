# CodeCoverage.cmake - CMake module for code coverage support

# Check for coverage tools
find_program(GCOV_PATH gcov)
find_program(LCOV_PATH lcov)
find_program(GENHTML_PATH genhtml)

if(NOT GCOV_PATH)
    message(STATUS "gcov not found, coverage disabled")
    return()
endif()

if(NOT LCOV_PATH)
    message(STATUS "lcov not found, coverage disabled")
    return()
endif()

# Coverage compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -fprofile-arcs -ftest-coverage")

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
