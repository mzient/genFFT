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

#ifndef GENFFT_TEST_REFERENCE_H
#define GENFFT_TEST_REFERENCE_H

#include <cassert>
#include <cmath>
#include <complex>
#include <vector>

namespace reference_impl {

template <typename T>
void FFT_pow2_inplace(std::complex<T> *out, int n, bool inv, int repeat = 1)
{
    if (n == 2)
    {
        for (int i = 0; i < 2*repeat; i += 2)
        {
            std::complex<T> a = out[i] + out[i+1];
            std::complex<T> b = out[i] - out[i+1];
            out[i] = a;
            out[i+1] = b;
        }
    }
    else if (n > 1)
    {
        int halfn = n >> 1;
        FFT_pow2_inplace(out, halfn, inv, repeat * 2);

        std::vector<std::complex<T>> twiddle(halfn);
        double da = 2 * M_PI / n;
        if (!inv) da = -da;
        double a = 0;
        for (int i = 0; i < halfn; i++, a += da)
        {
            twiddle[i] = { T(std::cos(a)), T(std::sin(a)) };
        }

        for (int k = 0; k < repeat; k++)
        {
            for (int i = 0; i < halfn; i++)
            {
                auto tmp = out[k * n + i + halfn] * twiddle[i];
                auto e = out[k * n + i] + tmp;
                auto o = out[k * n + i] - tmp;
                out[k * n + i] = e;
                out[k * n + i + halfn] = o;
            }
        }
    }
}

template <typename T, typename U>
void scramble(T *out, const U *in, int N, int stride = 1)
{
    if (N == 1)
        *out = *in;
    else
    {
        scramble<T, U>(out,          in,            N >> 1, stride << 1);
        scramble<T, U>(out + stride, in + (N >> 1), N >> 1, stride << 1);
    }
}

template <typename T, typename U>
void FFT_pow2(std::complex<T> *out, const U *in, int n, bool inv)
{
    assert((n & (n-1)) == 0);
    scramble(out, in, n);
    FFT_pow2_inplace(out, n, inv);
}

template <typename T>
void DFT(std::complex<T> *out, const std::complex<T> *in, int n, bool inv)
{
    double den = inv ? 2*M_PI / n : -2*M_PI / n;
    for (int i = 0; i < n; i++)
    {
        std::complex<double> c = 0;
        double a = 0, da = (double)i * den;
        for (int j = 0; j < n; j++, a += da)
        {
            std::complex<double> t = { std::cos(a), std::sin(a) };
            c += in[j] * t;
        }
        out[i] = { T(c.real()), T(c.imag()) };
    }
}

} // reference_impl

#endif // GENFFT_TEST_REFERENCE_H
