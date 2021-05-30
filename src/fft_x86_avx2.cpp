#include <cassert>
#include <immintrin.h>
#include <genFFT/FFTLevel.h>
#include <genFFT/generic/fft_impl_generic.h>
#include "dispatch_helper.h"

namespace genfft {
IMPORT_NAMESPACE(impl_AVX_FMA)
namespace impl_AVX2 {
    #include <genFFT/x86/fft_impl_x86.inl>

    FORWARD(float,  impl_AVX_FMA)
    FORWARD(double, impl_AVX_FMA)
}
}
