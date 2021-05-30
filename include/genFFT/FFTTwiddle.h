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

#ifndef GEN_FFT_TWIDDLE_H
#define GEN_FFT_TWIDDLE_H

#include <stdlib.h>
#include <cmath>
#include <utility>
#include "FFTAlloc.h"

namespace genfft {

/// @brief Twiddling factors for FFT
template <int _N, class T>
struct Twiddle
{
    static constexpr int N = _N;
    constexpr inline T operator[](int i) const noexcept { return t[i]; }
    alignas(32) T t[N];
    Twiddle()
    {
        for (int i=0; i<N; i+=2)
        {
            t[i]   =  std::cos(M_PI*i/N);
            t[i+1] = -std::sin(M_PI*i/N);
        }
    }
};

/// @brief Twiddling factors for FFT
template <class T>
struct Twiddle<-1, T>
{
    constexpr inline T operator[](int i) const noexcept { return t[i]; }
    T *t = nullptr;
    int N = 0;

    Twiddle() = default;
    explicit Twiddle(int n) : N(n)
    {
        t = aligned_alloc_T<T>(N, 32);
        for (int i=0; i<N; i+=2)
        {
            t[i]   =  std::cos(M_PI*i/N);
            t[i+1] = -std::sin(M_PI*i/N);
        }
    }

    Twiddle(const Twiddle &) = delete;
    Twiddle(Twiddle &&other) : t(other.t), N(other.N)
    {
        other.t = nullptr;
        other.N = 0;
    }

    Twiddle &operator=(const Twiddle &) = delete;
    Twiddle &operator=(Twiddle &other)
    {
        std::swap(t, other.t);
        std::swap(N, other.N);
        return *this;
    }

    ~Twiddle()
    {
        free(t);
        t = nullptr;
    }
};


/// @brief Twiddling factors for Decimation in Time (real input FFT)
template <int _N, typename T>
struct DITTwiddle
{
    static constexpr int N = _N;
    constexpr T operator[](int i) const noexcept { return t[i]; }
    static constexpr int pad = 32/sizeof(T) < 2 ? 2 : 32/sizeof(T);
    alignas(32) T t[N/2+pad];

    DITTwiddle()
    {
        Init();
    }


    explicit DITTwiddle(int n)
    {
        assert(n == N);
        Init();
    }

    void Init()
    {
        int i;
        for (i=0; i<=N/2; i+=2)
        {
            t[i] = std::sin(M_PI*i/N);
            t[i+1] = std::cos(M_PI*i/N);
        }
        for (; i<N/2+pad; i++)
            t[i] = 0;
    }
};

/// @brief Twiddling factors for Decimation in Time (real input FFT)
template <typename T>
struct DITTwiddle<-1, T>
{
    constexpr const T &operator[](int i) const noexcept { return t[i]; }
    int N = 0;
    T *t = nullptr;
    static constexpr int pad = 32/sizeof(T) < 2 ? 2 : 32/sizeof(T);

    DITTwiddle() = default;
    explicit DITTwiddle(int n)
    {
        Init(n);
    }

    void Init(int n)
    {
        if (n == N)
            return;
        free(t);
        N = n;
        if (N == 0)
            return;
        t = aligned_alloc_T<T>(N/2+pad, 32);
        int i;
        for (i=0; i<=N/2; i+=2)
        {
            t[i] = std::sin(M_PI*i/N);
            t[i+1] = std::cos(M_PI*i/N);
        }
        for (; i<N/2+pad; i++)
            t[i] = 0;
    }

    DITTwiddle(const DITTwiddle &) = delete;
    DITTwiddle(DITTwiddle &&other) : t(other.t), N(other.N)
    {
        other.t = nullptr;
        other.N = 0;
    }

    DITTwiddle &operator=(const DITTwiddle &) = delete;
    DITTwiddle &operator=(DITTwiddle &&other)
    {
        std::swap(t, other.t);
        std::swap(N, other.N);
        return *this;
    }

    ~DITTwiddle()
    {
        free(t);
        t = nullptr;
    }

};

} // genfft

#endif /* GEN_FFT_TWIDDLE_H */
