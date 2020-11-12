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

#ifndef GEN_FFT_LEVEL_H
#define GEN_FFT_LEVEL_H

#if (defined(_OPENMP) && (_OPENMP >= 201307L)) || (defined(_OPENMP_SIMD) && (_OPENMP_SIMD >= 201307L))
#define FFT_OPENMP_SIMD
#endif

#include "FFTUtil.h"
#include "FFTAlloc.h"

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

template <int N, class T, class Impl>
struct FFTLevel : FFTBase<T>, Impl
{
    void *operator new(size_t count)
    {
        return aligned_alloc_raw(count, alignof(FFTLevel));
    }
    void operator delete(void *p)
    {
        free(p);
    }

    void forward(T *data) override { Impl::template transform_impl<false>(data); }
    void inverse(T *data) override { Impl::template transform_impl<true>(data); }

    static std::shared_ptr<FFTBase<T>> GetInstance()
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
    static std::weak_ptr<FFTBase<T>> instance;
};

template <int N, class T, class Impl>
std::weak_ptr<FFTBase<T>> FFTLevel<N, T, Impl>::instance;

// Vertical multi-column FFT

template <class T>
inline int convenient_col_num(int cols)
{
    return cols;
}

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


template <int N, class T, class Impl>
struct FFTVertLevel : FFTVertBase<T>, Impl
{
    void *operator new(size_t count)
    {
        return aligned_alloc_raw(count, alignof(FFTVertLevel));
    }
    void operator delete(void *p)
    {
        free(p);
    }

    void forward(T *data, int stride, int columns) override { return Impl::template transform_impl<false>(data, stride, columns); }
    void inverse(T *data, int stride, int columns) override { return Impl::template transform_impl<true>(data, stride, columns); }

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

template <int N, class T, class Impl>
std::weak_ptr<FFTVertLevel<N, T, Impl>> FFTVertLevel<N, T, Impl>::instance;

} // impl
} // genfft

#endif /* GEN_FFT_LEVEL_H */

