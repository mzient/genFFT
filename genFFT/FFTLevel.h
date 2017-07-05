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

#ifndef GEN_FFT_LEVEL_H
#define GEN_FFT_LEVEL_H

#if (defined(_OPENMP) && (_OPENMP >= 201307L)) || (defined(_OPENMP_SIMD) && (_OPENMP_SIMD >= 201307L))
#define FFT_OPENMP_SIMD
#endif

#include "FFTUtil.h"
#include "FFTTwiddle.h"

#include <memory>

namespace genfft {
namespace impl {

// Single-row FFT
template <class T>
struct FFTBase
{
    virtual ~FFTBase() {}

    template <bool inv>
    inline void transform(T *data)
    {
        if (inv)
            inverse(data);
        else
            forward(data);
    }

    virtual void forward(T *)=0;
    virtual void inverse(T *)=0;
};

template <int N, class T>
struct FFTImplSelector;

template <int N, class T>
struct FFTImpl : FFTBase<T>, FFTImplSelector<N, T>::type
{
    typedef typename FFTImplSelector<N, T>::type Impl;
    void forward(T *data) override { Impl::template transform_impl<false>(data); }
    void inverse(T *data) override { Impl::template transform_impl<true>(data); }
};

template <int N, class T>
struct FFTLevel;

template <int N, class T>
struct FFTGeneric
{
    FFTLevel<N/2, T> next;
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
struct FFTImplSelector
{
    typedef FFTGeneric<N, T> type;
};

template <int N, class T>
struct FFTLevel : FFTImpl<N, T>
{
    void *operator new(size_t count)
    {
        return aligned_alloc(alignof(FFTLevel), count*sizeof(FFTLevel));
    }
    void operator delete(void *p)
    {
        free(p);
    }


    static std::shared_ptr<FFTLevel> GetInstance()
    {
        auto shptr = instance.lock();
        if (!shptr)
        {
            shptr.reset(new FFTLevel());
            instance = shptr;
        }
        return shptr;
    }

private:
    static std::weak_ptr<FFTLevel> instance;
};

template <int N, class T>
std::weak_ptr<FFTLevel<N, T>> FFTLevel<N, T>::instance;

// Vertical multi-column FFT

template <class T>
struct FFTVertBase
{
    virtual ~FFTVertBase() {}

    template <bool inv>
    inline void transform(T *data, int stride, int columns)
    {
        if (inv)
            inverse(data, stride, columns);
        else
            forward(data, stride, columns);
    }

    virtual void forward(T *data, int stride, int columns)=0;
    virtual void inverse(T *data, int stride, int columns)=0;
};

template <int N, class T>
struct FFTVertImplSelector;

template <int N, class T>
struct FFTVertImpl : FFTVertBase<T>, FFTVertImplSelector<N, T>::type
{
    typedef typename FFTVertImplSelector<N, T>::type Impl;
    void forward(T *data, int stride, int columns) override { return Impl::template transform_impl<false>(data, stride, columns); }
    void inverse(T *data, int stride, int columns) override { return Impl::template transform_impl<true>(data, stride, columns); }
};


template <int N, class T>
struct FFTVertLevel;

template <int N, class T>
struct FFTVertGeneric
{
    FFTVertLevel<N/2, T> next;

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

template <int N, class T>
struct FFTVertImplSelector
{
    typedef FFTVertGeneric<N, T> type;
};


template <int N, class T>
struct FFTVertLevel : FFTVertImpl<N, T>
{
    void *operator new(size_t count)
    {
        return aligned_alloc(alignof(FFTVertLevel), count*sizeof(FFTVertLevel));
    }
    void operator delete(void *p)
    {
        free(p);
    }

    static std::shared_ptr<FFTVertLevel> GetInstance()
    {
        auto shptr = instance.lock();
        if (!shptr)
        {
            shptr.reset(new FFTVertLevel());
            instance = shptr;
        }
        return shptr;
    }

private:
    static std::weak_ptr<FFTVertLevel> instance;
};

template <int N, class T>
std::weak_ptr<FFTVertLevel<N, T>> FFTVertLevel<N, T>::instance;

} // impl
} // genfft

#endif /* GEN_FFT_LEVEL_H */

