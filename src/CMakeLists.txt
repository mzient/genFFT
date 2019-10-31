project(genFFT_rt)

set(SOURCES
    fft_x86_dispatch.cpp
    x86_features.cpp
)

add_compile_options(-mavx2 -mavx -mfma -msse4.1 -mssse3 -msse3 -msse2 -msse)

add_library(genFFT_rt STATIC ${SOURCES})
target_link_libraries(genFFT_rt genFFT)
