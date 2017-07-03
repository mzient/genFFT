/*
Copyright 2017 Michal Zientkiewicz

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

#include "FFTComplex.h"
#include "FFTLevel.h"
#include "FFTFloat.h"
#include "FFTDouble.h"
#include <complex>
#include <cassert>

///@brief GenFFT - generic FFT
namespace genfft {

///@brief A 1D FFT for densely packed data
///@typeparam T scalar type
template <class T>
struct FFT
{
    FFT()=default;
    FFT(int n)
    {
        switch (n)
        {
#define SELECT_FFT_LEVEL(x) case (1<<x): impl = impl::FFTLevel<(1<<x), T>::GetInstance(); break;
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

    ///@brief Computes forward transform in-place, without data reordering
    ///@param inout data array
    void forward_no_scramble(std::complex<T> *inout)
    {
        impl->forward((T*)inout);
    }
    ///@brief Computes inverse transform in-place, without data reordering
    ///@param inout data array
    void inverse_no_scramble(std::complex<T> *inout)
    {
        impl->inverse((T*)inout);
    }

    ///@brief Computes forward transform
    ///@param out output array
    ///@param out input array
    void forward(std::complex<T> *out, const std::complex<T> *in)
    {
        scramble((complex<T>*)out, (const complex<T>*)in, n);
        impl->forward((T*)out);
    }

    ///@brief Computes inverse transform
    ///@param out output array
    ///@param out input array
    void inverse(std::complex<T> *out, const std::complex<T> *in)
    {
        scramble((complex<T>*)out, (const complex<T>*)in, n);
        impl->inverse((T*)out);
    }

    int size() const { return n; }

private:
    int n = 0;
    std::shared_ptr<impl::FFTBase<T>> impl;
};

///@brief Column-wise 1D FFT for multiple columns
///@typeparam T scalar type
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

    ///@brief Computes forward transform without data reordering
    ///@param data data array
    ///@param stride row stride, in complex elements, of the data array
    ///@param cols row length
    void forward_no_scramble(std::complex<T> *data, int stride, int cols)
    {
        impl->forward((T*)data, 2*stride, cols);
    }

    ///@brief Computes inverse transform without data reordering
    ///@param data data array
    ///@param stride row stride, in complex elements, of the data array
    ///@param cols row length
    void inverse_no_scramble(std::complex<T> *data, int stride, int cols)
    {
        impl->inverse((T*)data, 2*stride, cols);
    }

    ///@brief Computes forward transform
    ///@param out output array, must not be equal to in
    ///@param out_stride stride, in complex elements, of the output array
    ///@param in input array
    ///@param in_stride stride, in complex elements, of the input array
    ///@param cols row length
    void forward(std::complex<T> *out, int out_stride, const std::complex<T> *in, int in_stride, int cols)
    {
        struct cpx { T r, i; };
        scramble_rows((cpx*)out, out_stride, (const cpx*)in, in_stride, n, cols);
        impl->forward(out, out_stride, cols);
    }

    ///@brief Computes inverse transform
    ///@param out output array, must not be equal to in
    ///@param out_stride stride, in complex elements, of the output array
    ///@param in input array
    ///@param in_stride stride, in complex elements, of the input array
    ///@param cols row length
    void inverse(std::complex<T> *out, int out_stride, const std::complex<T> *in, int in_stride, int cols)
    {
        struct cpx { T r, i; };
        scramble_rows((cpx*)out, out_stride, (const cpx*)in, in_stride, n, cols);
        impl->inverse(out, out_stride, cols);
    }

    int size() const { return n; }

private:
    int n = 0;
    std::shared_ptr<impl::FFTVertBase<T>> impl;
};

///@brief 2D FFT
///@typeparam T scalar type
template <class T>
class FFT2D
{
public:
    FFT2D() = default;
    FFT2D(int width, int height) : horz(width), vert(height) {}

    ///@brief Computes forward transform
    ///@param out output array, must not be equal to in
    ///@param out_stride stride, in complex elements, of the output array
    ///@param in input array
    ///@param in_stride stride, in complex elements, of the input array
    void forward(std::complex<T> *out, int out_stride, const std::complex<T> *in, int in_stride)
    {
        scramble_row_fft<false>(out, out_stride, in, in_stride, rows());
        vert.forward_no_scramble(out, out_stride, cols());
    }

    ///@brief Computes inverse transform
    ///@param out output array, must not be equal to in
    ///@param out_stride stride, in complex elements, of the output array
    ///@param in input array
    ///@param in_stride stride, in complex elements, of the input array
    void inverse(std::complex<T> *out, int out_stride, const std::complex<T> *in, int in_stride)
    {
        scramble_row_fft<true>(out, out_stride, in, in_stride, rows());
        vert.inverse_no_scramble(out, out_stride, cols());
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
            if (inv)
                horz.inverse(out, in);
            else
                horz.forward(out, in);
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


#endif /* GENFFT_FFT_H */
