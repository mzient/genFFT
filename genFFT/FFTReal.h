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

#ifndef GEN_FFT_REAL_H
#define GEN_FFT_REAL_H

#include "fft.h"

namespace genfft {

///@brief 2D FFT
///@typeparam T scalar type
template <class T>
class FFT2D_real
{
public:
    FFT2D_real() = default;
    FFT2D_real(int width, int height) : horz(width), vert(height) {}

    ///@brief Computes forward transform
    ///@param out output array, must not be equal to in
    ///@param out_stride stride, in complex elements, of the output array
    ///@param in input array
    ///@param in_stride stride, in scalar elements, of the input array
    void forward(std::complex<T> *out, int out_stride, const T *in, int in_stride)
    {
        scramble_row_fwd(out, out_stride, in, in_stride, rows());
        vert.template transform_no_scramble<false>(out, out_stride, cols());
    }


    ///@brief Computes inverse transform
    ///@param out output array, must not be equal to in
    ///@param out_stride stride, in scalar elements, of the output array
    ///@param in input array
    ///@param in_stride stride, in complex elements, of the input array
    void inverse(T *out, int out_stride, const std::complex<T> *in, int in_stride)
    {
        assert(!"TODO");
    }

    ///@brief Number of columns in the domain
    int cols() const { return horz.size(); }
    ///@brief Number of rows in the domain
    int rows() const { return vert.size(); }

private:

    void scramble_row_fwd(std::complex<T> *out, int out_stride, const T *in, int in_stride, int rows)
    {
        if (rows == 1)
        {
            scramble(out, in, cols());
            horz.template transform_no_scramble<false>(out);
        }
        else
        {
            scramble_row_fwd(out,             2*out_stride, in,                    in_stride, rows/2);
            scramble_row_fwd(out+out_stride,  2*out_stride, in+(rows/2)*in_stride, in_stride, rows/2);
        }
    }

    FFT<T>          horz;
    FFTVert<T>      vert;
};

} // genFFT

#endif /* FFTREAL_H */

