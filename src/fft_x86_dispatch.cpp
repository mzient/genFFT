#include "x86_features.h"
#include <cassert>
#include <genFFT/FFTLevel.h>


namespace genfft {
namespace impl_generic {
    #include "fft_get_impl_proto.inl"
}

namespace impl_SSE {
    #include "fft_get_impl_proto.inl"
}

namespace impl_SSE2 {
    #include "fft_get_impl_proto.inl"
}

namespace impl_SSE3 {
    #include "fft_get_impl_proto.inl"
}

namespace impl_SSE41 {
    #include "fft_get_impl_proto.inl"
}

namespace impl_AVX {
    #include "fft_get_impl_proto.inl"
}

namespace impl_AVX_FMA {
    #include "fft_get_impl_proto.inl"
}

namespace impl_AVX2 {
    #include "fft_get_impl_proto.inl"
}

namespace impl_x86_dispatch {

template <typename T>
std::shared_ptr<impl::FFTBase<T>> GetImpl(int n, cpu_features cpu)
{
    if (cpu.AVX2) {
        return impl_AVX2::GetDispatchImpl(n, T());
    } else if (cpu.AVX) {
        if (cpu.FMA)
            return impl_AVX_FMA::GetDispatchImpl(n, T());
        else
            return impl_AVX::GetDispatchImpl(n, T());
    } else if (cpu.SSE41) {
        return impl_SSE41::GetDispatchImpl(n, T());
    } else if (cpu.SSE3) {
        return impl_SSE3::GetDispatchImpl(n, T());
    } else if (cpu.SSE2) {
        return impl_SSE3::GetDispatchImpl(n, T());
    } else if (cpu.SSE) {
        return impl_SSE::GetDispatchImpl(n, T());
    } else {
        return impl_generic::GetDispatchImpl(n, T());
    }
}

std::shared_ptr<impl::FFTBase<float>> GetImpl(int n, float)
{
    return GetImpl<float>(n, GetCPUFeatures());
}

std::shared_ptr<impl::FFTBase<double>> GetImpl(int n, double)
{
    return GetImpl<double>(n, GetCPUFeatures());
}

template <typename T>
std::shared_ptr<impl::FFTVertBase<T>> GetVertImpl(int n, cpu_features cpu)
{
    if (cpu.AVX2) {
        return impl_AVX2::GetVertDispatchImpl(n, T());
    } else if (cpu.AVX) {
        if (cpu.FMA)
            return impl_AVX_FMA::GetVertDispatchImpl(n, T());
        else
            return impl_AVX::GetVertDispatchImpl(n, T());
    } else if (cpu.SSE41) {
        return impl_SSE41::GetVertDispatchImpl(n, T());
    } else if (cpu.SSE3) {
        return impl_SSE3::GetVertDispatchImpl(n, T());
    } else if (cpu.SSE2) {
        return impl_SSE3::GetVertDispatchImpl(n, T());
    } else if (cpu.SSE) {
        return impl_SSE::GetVertDispatchImpl(n, T());
    } else {
        return impl_generic::GetVertDispatchImpl(n, T());
    }
}

std::shared_ptr<impl::FFTVertBase<float>> GetVertImpl(int n, float)
{
    return GetVertImpl<float>(n, GetCPUFeatures());
}

std::shared_ptr<impl::FFTVertBase<double>> GetVertImpl(int n, double)
{
    return GetVertImpl<double>(n, GetCPUFeatures());
}

}  // impl_x86_dispatch

} // genFFT
