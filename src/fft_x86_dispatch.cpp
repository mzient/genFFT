#include "x86_features.h"
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <cassert>
#include <genFFT/FFTLevel.h>
#include <genFFT/generic/fft_impl_generic.h>

namespace genfft {
namespace impl_SSE {
    #define GENFFT_USE_SSE
    #include <genFFT/x86/fft_float_impl_x86.inl>

    inline std::shared_ptr<impl::FFTBase<double>> GetImpl(int n, double) {
        return impl_generic::GetImpl(n, double());
    }
}

namespace impl_SSE2 {
    #define GENFFT_USE_SSE2
    #include <genFFT/x86/fft_double_impl_x86.inl>
    inline std::shared_ptr<impl::FFTBase<float>> GetImpl(int n, float) {
        return impl_SSE::GetImpl(n, float());
    }
}

namespace impl_SSE3 {
    #define GENFFT_USE_SSE3
    #include <genFFT/x86/fft_float_impl_x86.inl>
    #include <genFFT/x86/fft_double_impl_x86.inl>
}

namespace impl_SSE41 {
    #define GENFFT_USE_SSE4_1
    #include <genFFT/x86/fft_float_impl_x86.inl>
    #include <genFFT/x86/fft_double_impl_x86.inl>
}

namespace impl_AVX {
    #define GENFFT_USE_AVX
    #include <genFFT/x86/fft_float_impl_x86.inl>
    #include <genFFT/x86/fft_double_impl_x86.inl>
}

namespace impl_AVX_FMA {
    #define GENFFT_USE_FMA
    #include <genFFT/x86/fft_float_impl_x86.inl>
    #include <genFFT/x86/fft_double_impl_x86.inl>
}

namespace impl_AVX2 {
    #define USE_AVX2
    #include <genFFT/x86/fft_float_impl_x86.inl>

    inline std::shared_ptr<impl::FFTBase<double>> GetImpl(int n, double) {
        return impl_AVX_FMA::GetImpl(n, double());
    }
}

namespace impl_x86_dispatch {

template <typename T>
std::shared_ptr<impl::FFTBase<T>> GetImpl(int n)
{
    cpu_features cpu = GetCPUFeatures();
    if (cpu.AVX2) {
        return impl_AVX2::GetImpl(n, T());
    } else if (cpu.AVX) {
        if (cpu.FMA)
            return impl_AVX_FMA::GetImpl(n, T());
        else
            return impl_AVX::GetImpl(n, T());
    } else if (cpu.SSE41) {
        return impl_SSE41::GetImpl(n, T());
    } else if (cpu.SSE3) {
        return impl_SSE3::GetImpl(n, T());
    } else if (cpu.SSE2) {
        return impl_SSE3::GetImpl(n, T());
    } else if (cpu.SSE) {
        return impl_SSE::GetImpl(n, T());
    } else {
        return impl_generic::GetImpl(n, T());
    }
}

std::shared_ptr<impl::FFTBase<float>> GetImpl(int n, float)
{
    return GetImpl<float>(n);
}

std::shared_ptr<impl::FFTBase<double>> GetImpl(int n, double)
{
    return GetImpl<double>(n);
}

}  // impl_x86_dispatch

} // genFFT
