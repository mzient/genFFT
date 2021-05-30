#include <cassert>
#include <immintrin.h>
#include <genFFT/FFTLevel.h>
#include <genFFT/generic/fft_impl_generic.h>
#include "dispatch_helper.h"

namespace genfft {
namespace impl_AVX_FMA {
    #include <genFFT/x86/fft_impl_x86.inl>

    DISPATCH_ALL()
}
}
