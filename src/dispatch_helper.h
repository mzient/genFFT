#ifndef GENFFT_DISPATCH_HELPER_H
#define GENFFT_DISPATCH_HELPER_H

#include <genFFT/x86/fft_x86_preproc.h>

namespace genfft {
template <typename T>
using FFTImplPtr = std::shared_ptr<impl::FFTBase<T>>;
template <typename T>
using FFTVertImplPtr = std::shared_ptr<impl::FFTVertBase<T>>;

} // genfft

#define IMPORT_NAMESPACE(name) namespace name { \
std::shared_ptr<impl::FFTBase<float>> GetDispatchImpl(int n, float);            \
std::shared_ptr<impl::FFTVertBase<float>> GetVertDispatchImpl(int n, float);    \
std::shared_ptr<impl::FFTBase<double>> GetDispatchImpl(int n, double);          \
std::shared_ptr<impl::FFTVertBase<double>> GetVertDispatchImpl(int n, double);  \
}

#define DISPATCH(type)\
genfft::FFTImplPtr<type> GetDispatchImpl(int n, type dummy)  { return GetImpl(n, dummy); } \
genfft::FFTVertImplPtr<type> GetVertDispatchImpl(int n, type dummy)  { return GetVertImpl(n, dummy); } \

#define DISPATCH_ALL() DISPATCH(float) DISPATCH(double)

#define FORWARD(type, from_namespace)\
genfft::FFTImplPtr<type> GetDispatchImpl(int n, type dummy)  { return from_namespace::GetDispatchImpl(n, dummy); } \
genfft::FFTVertImplPtr<type> GetVertDispatchImpl(int n, type dummy)  { return from_namespace::GetVertDispatchImpl(n, dummy); } \

#endif
