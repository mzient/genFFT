#include <cassert>
#include <xmmintrin.h>
#include <genFFT/FFTLevel.h>
#include <genFFT/generic/fft_impl_generic.h>
#include "dispatch_helper.h"

namespace genfft {
namespace impl_generic {
    DISPATCH_ALL()
}
}
