/*
Copyright 2017-2020 Michal Zientkiewicz

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef GENFFT_FFT_H
#define GENFFT_FFT_H

#include "FFTTypes.h"
#include "FFTDecl.h"
#include "FFTLevel.h"
#include "FFTDIT.h"
#include "FFTBackend.h"
#include <complex>
#include <cassert>

///@brief GenFFT - generic FFT
namespace genfft {

template <typename T>
using FFTImplPtr = std::shared_ptr<impl::FFTBase<T>>;
template <typename T>
using FFTVertImplPtr = std::shared_ptr<impl::FFTVertBase<T>>;
template <typename T>
using FFTDITImplPtr = std::shared_ptr<impl::FFTDITBase<T>>;
template <typename T>
using FFTImplFactory = FFTImplPtr<T>(int n, T);
template <typename T>
using FFTVertImplFactory = FFTVertImplPtr<T>(int n, T);
template <typename T>
using FFTDITImplFactory = FFTDITImplPtr<T>(int n, T);

///@brief A 1D FFT for densely packed data
///@tparam T scalar type
template <class T, FFTImplFactory<T> *factory = backend::GetImpl>
struct FFT
{
    FFT()=default;
    explicit FFT(int n)
    {
        impl = factory(n, T());
        this->n = n;
    }

    ///@brief Computes transform in-place, without data reordering
    ///@tparam inv if true, computes inverse transform
    ///@param inout data array
    template <bool inv>
    void transform_no_scramble(std::complex<T> *inout)
    {
        impl->template transform<inv>((T*)inout);
    }


    ///@brief Computes transform
    ///@tparam inv if true, computes inverse transform
    ///@param out output array
    ///@param out input array
    template <bool inv>
    void transform(std::complex<T> *out, const std::complex<T> *in)
    {
        scramble((complex<T>*)out, (const complex<T>*)in, n);
        impl->template transform<inv>((T*)out);
    }

    ///@brief Computes forward transform of real data
    ///@param out output array
    ///@param out input array
    void transform_real(std::complex<T> *out, const T *in)
    {
        scramble(out, in, n);
        impl->template transform<false>((T*)out);
    }

    ///@brief Computes forward transform of interleaved real data.
    ///       Requires postprocessing step to separate the transforms.
    ///@param out output array
    ///@param out input array
    void transform_interleave(std::complex<T> *out, const T *in1, const T *in2)
    {
        scramble((T*)out,   in1, n, 2);
        scramble((T*)out+1, in2, n, 2);
        impl->template transform<false>((T*)out);
    }

    int size() const noexcept { return n; }
    explicit operator bool() const noexcept { return impl; }

private:
    int n = 0;
    FFTImplPtr<T> impl;
};

///@brief Column-wise 1D FFT for multiple columns
///@tparam T scalar type
template <class T, FFTVertImplFactory<T> *factory = backend::GetVertImpl>
struct FFTVert
{
    FFTVert()=default;
    explicit FFTVert(int n)
    {
        impl = factory(n, T());
        this->n = n;
    }

    ///@brief Computes transform without data reordering
    ///@tparam inv if true, computes inverse transform
    ///@param data data array
    ///@param stride row stride, in complex numbers, of the data array
    ///@param cols row length
    template <bool inv>
    void transform_no_scramble(std::complex<T> *data, stride_t stride, index_t cols)
    {
        impl->template transform<inv>((T*)data, 2*stride, cols);
    }

    ///@brief Computes transform
    ///@tparam inv if true, computes inverse transform
    ///@param out output array, must not be equal to in
    ///@param out_stride stride, in complex numbers, of the output array
    ///@param in input array
    ///@param in_stride stride, in complex numbers, of the input array
    ///@param cols row length
    template <bool inv>
    void transform(std::complex<T> *out, stride_t out_stride, const std::complex<T> *in, stride_t in_stride, index_t cols)
    {
        scramble_rows((complex<T>*)out, out_stride, (const complex<T>*)in, in_stride, n, cols);
        impl->template transform<inv>((T*)out, 2*out_stride, cols);
    }


    ///@brief Computes transform
    ///@tparam inv if true, computes inverse transform
    ///@param out output array, must not be equal to in
    ///@param in input array
    ///@param cols row length
    template <bool inv>
    void transform(std::complex<T> *out, const std::complex<T> *in, index_t cols)
    {
        scramble_rows((complex<T>*)out, cols, (const complex<T>*)in, cols, n, cols);
        impl->template transform<inv>((T*)out, 2*cols, cols);
    }

    int size() const noexcept { return n; }
    explicit operator bool() const noexcept { return impl; }

private:
    int n = 0;
    std::shared_ptr<impl::FFTVertBase<T>> impl;
};

template <class T, FFTDITImplFactory<T> *factory = backend::GetDITImpl>
class DIT
{
public:
    DIT() = default;
    explicit DIT(int n) : n(n), impl(factory(n, T())) {}

    void apply(T *out, const T *in, bool half)
    {
        impl->apply(out, in, half);
    }

    void apply(std::complex<T> *out, const std::complex<T> *in, bool half)
    {
        impl->apply((T*)out, (const T *)in, half);
    }

    int size() const noexcept { return n; }
    explicit operator bool() const noexcept { return impl; }

private:
    int n = 0;
    std::shared_ptr<impl::FFTDITBase<T>> impl;
};

///@brief 2D FFT
///@tparam T scalar type
template <class T>
class FFT2D
{
public:
    FFT2D() = default;
    FFT2D(int width, int height) : horz(width), vert(height) {}

    ///@brief Computes transform
    ///@tparam inv if true, computes inverse transform
    ///@param out output array, must not be equal to in
    ///@param out_stride stride, in complex elements, of the output array
    ///@param in input array
    ///@param in_stride stride, in complex elements, of the input array
    template <bool inv>
    void transform(std::complex<T> *out, stride_t out_stride, const std::complex<T> *in, stride_t in_stride)
    {
        scramble_row_fft<inv>(out, out_stride, in, in_stride, rows());
        vert.template transform_no_scramble<inv>(out, out_stride, cols());
    }

    ///@brief Number of columns in the domain
    int cols() const noexcept { return horz.size(); }
    ///@brief Number of rows in the domain
    int rows() const noexcept { return vert.size(); }

    explicit operator bool() const noexcept { return horz && vert; }

private:

    template <bool inv>
    void scramble_row_fft(std::complex<T> *out, stride_t out_stride, const std::complex<T> *in, stride_t in_stride, index_t rows)
    {
        if (rows == 1)
        {
            horz.template transform<inv>(out, in);
        }
        else
        {
            scramble_row_fft<inv>(out,             2*out_stride, in,                    in_stride, rows/2);
            scramble_row_fft<inv>(out+out_stride,  2*out_stride, in+(rows/2)*in_stride, in_stride, rows/2);
        }
    }

    FFT<T>      horz;
    FFTVert<T>  vert;
};

} // genfft

#include "FFTReal.h"


#endif /* GENFFT_FFT_H */
