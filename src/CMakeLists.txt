project(genFFT_rt)

set(SOURCES
    fft_x86_dispatch.cpp
    x86_features.cpp
    fft_generic.cpp
    fft_x86_sse.cpp
    fft_x86_sse2.cpp
    fft_x86_sse3.cpp
    fft_x86_sse41.cpp
    fft_x86_avx.cpp
    fft_x86_avx_fma.cpp
    fft_x86_avx2.cpp
)

#add_compile_options(-mavx2 -mavx -mfma -msse4.1 -mssse3 -msse3 -msse2 -msse)
set_source_files_properties(fft_x86_sse PROPERTIES COMPILE_FLAGS "-msse")
set_source_files_properties(fft_x86_sse2 PROPERTIES COMPILE_FLAGS "-msse2")
set_source_files_properties(fft_x86_sse3 PROPERTIES COMPILE_FLAGS "-msse3")
set_source_files_properties(fft_x86_sse41 PROPERTIES COMPILE_FLAGS "-msse4")
set_source_files_properties(fft_x86_avx PROPERTIES COMPILE_FLAGS "-msse3 -mavx")
set_source_files_properties(fft_x86_avx_fma PROPERTIES COMPILE_FLAGS "-msse3 -mavx -mfma")
set_source_files_properties(fft_x86_avx2 PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

add_library(genFFT_rt STATIC ${SOURCES})
target_link_libraries(genFFT_rt genFFT)
