find_package(GTest REQUIRED)
add_executable(genfft_test
    test_fft.cpp
    test_dispatch.cpp
    test_reference.cpp
    test_main.cpp)
target_link_libraries(genfft_test gtest genFFT_rt pthread)

find_package(benchmark REQUIRED)
add_executable(genfft_bench fft_bench.cpp)
target_link_libraries(genfft_bench benchmark genFFT_rt pthread)
