#include <cassert>
#include <emmintrin.h>
#include <genFFT/FFTLevel.h>
#include <genFFT/generic/fft_impl_generic.h>
#include "dispatch_helper.h"

namespace genfft {
IMPORT_NAMESPACE(impl_SSE)
namespace impl_SSE2 {
    #include <genFFT/x86/fft_double_impl_x86.inl>

    DISPATCH(double)
    FORWARD(float, impl_SSE)
}
}