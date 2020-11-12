cmake_minimum_required(VERSION 3.9)
project(genFFT)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

add_library(genFFT INTERFACE)
target_include_directories(genFFT INTERFACE "include/")

add_subdirectory(src)

if (${BUILD_TESTS})
    message("Build with tests")
    add_subdirectory(test)
endif()
