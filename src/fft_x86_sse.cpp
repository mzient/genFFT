#include <cassert>
#include <xmmintrin.h>
#include <genFFT/FFTLevel.h>
#include <genFFT/generic/fft_impl_generic.h>
#include "dispatch_helper.h"

namespace genfft {
IMPORT_NAMESPACE(impl_generic)
namespace impl_SSE {
    #include <genFFT/x86/fft_float_impl_x86.inl>

    DISPATCH(float)
    FORWARD(double, impl_generic)
}
}
