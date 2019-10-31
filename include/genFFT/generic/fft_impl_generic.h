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

#ifndef GENFFT_IMPL_GENERIC_H
#define GENFFT_IMPL_GENERIC_H

#include <cassert>
#include "../FFTTwiddle.h"

namespace genfft {
namespace impl_generic {

template <int N, class T>
struct FFTGeneric
{
    FFTGeneric<N/2, T> next;
    template <bool inv>
    void transform_impl(T *data)
    {
        next.template transform_impl<inv>(data);
        next.template transform_impl<inv>(data+N);

        T tempr, tempi;

#ifdef FFT_OPENMP_SIMD
        #pragma omp simd
#endif
        for (unsigned i=0; i<N; i+=2)
        {
            T wr = twiddle[i];
            T wi = twiddle[i+1];
            tempr = inv ? data[i+N]*wr + data[i+N+1]*wi : data[i+N]*wr - data[i+N+1]*wi;
            tempi = inv ? data[i+N+1]*wr - data[i+N]*wi : data[i+N]*wi + data[i+N+1]*wr;
            data[i+N]   = data[i]-tempr;
            data[i+N+1] = data[i+1]-tempi;
            data[i]     += tempr;
            data[i+1]   += tempi;
        }
    }

    const Twiddle<N, T> twiddle;
};


template <class T>
struct FFTGeneric<4, T>
{
    template <bool inv>
    void transform_impl(T *data)
    {
        T tr = data[2];
        T ti = data[3];
        data[2] = data[0]-tr;
        data[3] = data[1]-ti;
        data[0] += tr;
        data[1] += ti;
        tr = data[6];
        ti = data[7];
        data[6] = inv ? ti-data[5] : data[5]-ti;
        data[7] = inv ? data[4]-tr : tr-data[4];
        data[4] += tr;
        data[5] += ti;

        tr = data[4];
        ti = data[5];
        data[4] = data[0]-tr;
        data[5] = data[1]-ti;
        data[0] += tr;
        data[1] += ti;
        tr = data[6];
        ti = data[7];
        data[6] = data[2]-tr;
        data[7] = data[3]-ti;
        data[2] += tr;
        data[3] += ti;
    }
};

template <class T>
struct FFTGeneric<2, T>
{
    template <bool inv_unused>
    void transform_impl(T *data)
    {
        T tr = data[2];
        T ti = data[3];
        data[2] = data[0]-tr;
        data[3] = data[1]-ti;
        data[0] += tr;
        data[1] += ti;
    }
};

template <class T>
struct FFTGeneric<1, T>
{
    template <bool inv_unused>
    void transform_impl(T *) {}
};



template <int N, class T>
struct FFTVertGeneric
{
    FFTVertGeneric<N/2, T> next;

    template <bool inv>
    void transform_impl(T *data, int stride, int columns)
    {
        const int half = N/2*stride;
        next.template transform_impl<inv>(data,      stride, columns);
        next.template transform_impl<inv>(data+half, stride, columns);

        for (unsigned i=0; i<N/2; i++)
        {
            T wr = twiddle[2*i];
            T wi = twiddle[2*i+1];
            T *even = data + i*stride;
            T *odd =  even + half;
#ifdef FFT_OPENMP_SIMD
            #pragma omp simd
#endif
            for (int j=0; j<2*columns; j+=2)
            {
                T tempr, tempi;
                tempr = inv ? odd[j]*wr + odd[j+1]*wi : odd[j]*wr - odd[j+1]*wi;
                tempi = inv ? odd[j+1]*wr - odd[j]*wi : odd[j+1]*wr + odd[j]*wi;
                odd[j]    = even[j]-tempr;
                odd[j+1]  = even[j+1]-tempi;
                even[j]   += tempr;
                even[j+1] += tempi;
            }
        }
    }

    const Twiddle<N, T> twiddle;
};


template <class T>
struct FFTVertGeneric<4, T>
{
    template <bool inv>
    void transform_impl(T *data, int stride, int cols)
    {
        T *row0 = data;
        T *row1 = row0+stride;
        T *row2 = row1+stride;
        T *row3 = row2+stride;

#ifdef FFT_OPENMP_SIMD
        #pragma omp simd
#endif
        for (int j=0; j<cols*2; j+=2)
        {
            T tr = row1[j];
            T ti = row1[j+1];
            row1[j]     = row0[j]-tr;
            row1[j+1]   = row0[j+1]-ti;
            row0[j]     += tr;
            row0[j+1]   += ti;
            tr = row3[j];
            ti = row3[j+1];
            row3[j]     = inv ? ti-row2[j+1] : row2[j+1]-ti;
            row3[j+1]   = inv ? row2[j]-tr : tr-row2[j];
            row2[j]     += tr;
            row2[j+1]   += ti;

            tr = row2[j];
            ti = row2[j+1];
            row2[j]     = row0[j]-tr;
            row2[j+1]   = row0[j+1]-ti;
            row0[j]     += tr;
            row0[j+1]   += ti;
            tr = row3[j];
            ti = row3[j+1];
            row3[j]     = row1[j]-tr;
            row3[j+1]   = row1[j+1]-ti;
            row1[j]     += tr;
            row1[j+1]   += ti;
        }
    }
};

template <class T>
struct FFTVertGeneric<2, T>
{
    template <bool inv_unused>
    void transform_impl(T* data, int stride, int cols)
    {
        T* row0 = data;
        T* row1 = data+stride;

        for (int i=0; i<2*cols; i+=2)
        {
            T tr = row1[i];
            T ti = row1[i+1];
            row1[i  ] = row0[i  ]-tr;
            row1[i+1] = row0[i+1]-ti;
            row0[i  ] += tr;
            row0[i+1] += ti;
        }
    }

};

template <class T>
struct FFTVertGeneric<1, T>
{
    template <bool inv_unused>
    void transform_impl(T*, int, int) {}
};

template <class T>
inline std::shared_ptr<impl::FFTBase<T>> GetImpl(int n, T)
{
    switch (n) {
#define SELECT_FFT_LEVEL(x) case (1<<x): return impl::FFTLevel<(1<<x), T, FFTGeneric<(1<<x), T>>::GetInstance();
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
}

}  // impl_generic
}  // genfft

#endif /* GENFFT_IMPL_GENERIC_H */
