/*
Copyright 2017-2019 Michal Zientkiewicz

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

#include "FFTDecl.h"
#include "FFTLevel.h"
#include "FFTBackend.h"
#include <complex>
#include <cassert>

///@brief GenFFT - generic FFT
namespace genfft {

template <typename T>
using FFTImplPtr = std::shared_ptr<impl::FFTBase<T>>;
template <typename T>
using FFTImplFactory = FFTImplPtr<T>(int n, T);

///@brief A 1D FFT for densely packed data
///@tparam T scalar type
template <class T, FFTImplFactory<T> *factory = backend::GetImpl>
struct FFT
{
    FFT()=default;
    FFT(int n)
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

    int size() const { return n; }

private:
    int n = 0;
    FFTImplPtr<T> impl;
};

///@brief Column-wise 1D FFT for multiple columns
///@tparam T scalar type
template <class T>
struct FFTVert
{
    FFTVert()=default;
    FFTVert(int n)
    {
        switch (n)
        {
#define SELECT_FFT_LEVEL(x) case (1<<x): impl = impl::FFTVertLevel<(1<<x), T>::GetInstance(); break;
            SELECT_FFT_LEVEL(0);
            SELECT_FFT_LEVEL(1);
            SELECT_FFT_LEVEL(2);
            SELECT_FFT_LEVEL(3);
            SELECT_FFT_LEVEL(4);
            SELECT_FFT_LEVEL(5);
            SELECT_FFT_LEVEL(6);
            SELECT_FFT_LEVEL(7);
            SELECT_FFT_LEVEL(8);
            SELECT_FFT_LEVEL(9);
            SELECT_FFT_LEVEL(10);
            SELECT_FFT_LEVEL(11);
            SELECT_FFT_LEVEL(12);
            SELECT_FFT_LEVEL(13);
            SELECT_FFT_LEVEL(14);
            SELECT_FFT_LEVEL(15);
            SELECT_FFT_LEVEL(16);
            SELECT_FFT_LEVEL(17);
            SELECT_FFT_LEVEL(18);
            SELECT_FFT_LEVEL(19);
            SELECT_FFT_LEVEL(20);
            SELECT_FFT_LEVEL(21);
            SELECT_FFT_LEVEL(22);
            SELECT_FFT_LEVEL(23);
#undef SELECT_FFT_LEVEL
        default:
            assert(!"unsupported size");
        }
        this->n = n;
    }

    ///@brief Computes transform without data reordering
    ///@tparam inv if true, computes inverse transform
    ///@param data data array
    ///@param stride row stride, in complex elements, of the data array
    ///@param cols row length
    template <bool inv>
    void transform_no_scramble(std::complex<T> *data, int stride, int cols)
    {
        impl->template transform<inv>((T*)data, 2*stride, cols);
    }

    ///@brief Computes transform
    ///@tparam inv if true, computes inverse transform
    ///@param out output array, must not be equal to in
    ///@param out_stride stride, in complex elements, of the output array
    ///@param in input array
    ///@param in_stride stride, in complex elements, of the input array
    ///@param cols row length
    template <bool inv>
    void transform(std::complex<T> *out, int out_stride, const std::complex<T> *in, int in_stride, int cols)
    {
        scramble_rows((complex<T>*)out, out_stride, (const complex<T>*)in, in_stride, n, cols);
        impl->template transform<inv>(out, out_stride, cols);
    }

    int size() const { return n; }

private:
    int n = 0;
    std::shared_ptr<impl::FFTVertBase<T>> impl;
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
    void transform(std::complex<T> *out, int out_stride, const std::complex<T> *in, int in_stride)
    {
        scramble_row_fft<inv>(out, out_stride, in, in_stride, rows());
        vert.template transform_no_scramble<inv>(out, out_stride, cols());
    }

    ///@brief Number of columns in the domain
    int cols() const { return horz.size(); }
    ///@brief Number of rows in the domain
    int rows() const { return vert.size(); }

private:

    template <bool inv>
    void scramble_row_fft(std::complex<T> *out, int out_stride, const std::complex<T> *in, int in_stride, int rows)
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
