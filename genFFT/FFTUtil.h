/*
Copyright (C) 2017 Michal Zientkiewicz

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

#ifndef GEN_FFT_UTIL_H
#define GEN_FFT_UTIL_H

#include <type_traits>

namespace genfft
{

template <int N, class T>
inline typename std::enable_if<(N == 1), void>::type static_scramble(T *out, const T *in, int stride = 1)
{
    (void)stride;
    *out = *in;
}

template <int N, class T>
inline typename std::enable_if<(N>1), void>::type static_scramble(T *out, const T *in, int stride = 1)
{
    static_scramble<N/2, T>(out, in, 2 * stride);
    static_scramble<N/2, T>(out + stride, in + N/2, 2 * stride);
}

template <class T>
void scramble(T *out, const T *in, int N, int stride = 1)
{
    switch (N)
    {
    case 0:
        break;
    case 1:
        static_scramble<1>(out, in, stride); break;
    case 2:
        static_scramble<2>(out, in, stride); break;
    case 4:
        static_scramble<4>(out, in, stride); break;
    case 8:
        static_scramble<8>(out, in, stride); break;
    case 16:
        static_scramble<16>(out, in, stride); break;
    case 32:
        static_scramble<32>(out, in, stride); break;
    case 64:
        static_scramble<64>(out, in, stride); break;
    case 128:
        static_scramble<128>(out, in, stride); break;
    case 256:
        static_scramble<256>(out, in, stride); break;
    case 512:
        static_scramble<512>(out, in, stride); break;
    default:
        scramble<T>(out,          in,       N/2, 2*stride);
        scramble<T>(out + stride, in + N/2, N/2, 2*stride);
        break;
    }
}

template <class T>
void scramble_rows(T *out, int out_stride, const T *in, int in_stride, int rows, int cols)
{
    if (rows == 1)
    {
        for (int i=0; i<cols; i++)
            out[i] = in[i];
    }
    else
    {
        scramble_rows<T>(out,             2*out_stride, in,                    in_stride, rows/2, cols);
        scramble_rows<T>(out+out_stride,  2*out_stride, in+(rows/2)*in_stride, in_stride, rows/2, cols);
    }
}

} // genfft

#endif /* GEN_FFT_UTIL_H */

