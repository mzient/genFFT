find_package(GTest REQUIRED)
add_executable(genfft_test
    test_fft.cpp
    test_real_fft.cpp
    test_dit.cpp
    test_dispatch.cpp
    test_reference.cpp
    test_x86.cpp
    test_main.cpp)
target_link_libraries(genfft_test gtest genFFT_rt pthread)

find_package(benchmark REQUIRED)
add_executable(genfft_bench fft_bench.cpp)
target_link_libraries(genfft_bench benchmark genFFT_rt pthread)

target_compile_options(genfft_bench PRIVATE -march=native)
target_compile_options(genfft_test  PRIVATE -march=native)
