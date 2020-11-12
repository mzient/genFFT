/*
Copyright 2019-2020 Michal Zientkiewicz

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

#ifndef GENFFT_TEST_TEST_UTIL_H
#define GENFFT_TEST_TEST_UTIL_H

#include <vector>
#include <complex>
#include <random>

#include "fft_ref_impl.h"

template <typename T>
void DummyData(std::vector<std::complex<T>> &vec, bool real)
{
    std::mt19937_64 rng;
    std::uniform_real_distribution<T> dist(-1, 1);

    for (auto &c : vec)
    {
        c.real(dist(rng));
        c.imag(real ? 0 : dist(rng));
    }
}

template <typename T>
void DummyData(std::vector<T> &vec)
{
    std::mt19937_64 rng;
    std::uniform_real_distribution<T> dist(-1, 1);

    for (auto &x : vec)
        x = dist(rng);
}

template <typename T>
constexpr double FFT_Eps(int n);

template <>
constexpr double FFT_Eps<double>(int n)
{
    return 1e-8 + n*1e-12;
}

template <>
constexpr double FFT_Eps<float>(int n)
{
    return 1e-5 + n*1e-8;
}

#endif // GENFFT_TEST_TEST_UTIL_H
