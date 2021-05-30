#include <cassert>
#include <smmintrin.h>
#include <genFFT/FFTLevel.h>
#include <genFFT/generic/fft_impl_generic.h>
#include "dispatch_helper.h"

namespace genfft {
namespace impl_SSE41 {
    #include <genFFT/x86/fft_impl_x86.inl>

    DISPATCH_ALL()
}
}
