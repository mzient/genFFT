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

/*///@brief Recovers two transforms of real data from one interleaved transform.
template <class T>
void separate_2x_real_FFT(std::complex<T> *out1, std::complex<T> *out2, const std::complex<T> *in, int N)
{
    const T *Fz = (const T*)in;
    T *Fx = (T *)out1;
    T *Fy = (T *)out2;

    Fx[0] = Fz[0];
    Fx[1] = 0;
    Fy[0] = Fz[1];
    Fy[1] = 0;

    for (int i=1; i<N; i++)
    {
        int k = N-i;
        Fx[2*i+0] = (Fz[2*i]   + Fz[2*k])   * 0.5f;
        Fx[2*i+1] = (Fz[2*i+1] - Fz[2*k+1]) * 0.5f;
        Fy[2*i+0] = (Fz[2*k+1] + Fz[2*i+1]) * 0.5f;
        Fy[2*i+1] = (Fz[2*k]   - Fz[2*i])   * 0.5f;
    }
}*/

///@brief Recovers two transforms of real data from one interleaved transform. The input and output arrays may alias.
template <class T>
void separate_2x_real_FFT(std::complex<T> *out1, std::complex<T> *out2, const std::complex<T> *in, int N)
{
    const T *Fz = (const T*)in;
    T *Fx = (T *)out1;
    T *Fy = (T *)out2;

    T xr, xi, yr, yi;
    xr = Fz[0];
    yr = Fz[1];
    Fx[0] = xr;
    Fx[1] = 0;
    Fy[0] = yr;
    Fy[1] = 0;

    for (int i=1; i<=N/2; i++)
    {
        int k = N-i;
        xr = (Fz[2*i]   + Fz[2*k])   * 0.5f;
        xi = (Fz[2*i+1] - Fz[2*k+1]) * 0.5f;
        yr = (Fz[2*k+1] + Fz[2*i+1]) * 0.5f;
        yi = (Fz[2*k]   - Fz[2*i])   * 0.5f;
        Fx[2*i+0] = xr;
        Fx[2*i+1] = xi;
        Fy[2*i+0] = yr;
        Fy[2*i+1] = yi;
        Fx[2*k+0] = xr;
        Fx[2*k+1] =-xi;
        Fy[2*k+0] = yr;
        Fy[2*k+1] =-yi;
    }
}


///@brief 2D FFT
///@tparam T scalar type
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
        const int M = cols();
        const int N = rows();
        scramble_row_fwd(out, out_stride, in, in_stride, N);
        const int vcols = vert_cols();
        vert.template transform_no_scramble<false>(out, out_stride, vcols);
        if (vcols != M)
        {
            for (int i=0; i<rows(); i++)
            {
                int k = i ? N-i : 0;
                T *irow = (T*)&out[i*out_stride];
                T *orow = (T*)&out[k*out_stride];
                for (int j=M-1; j>=vcols; j--)
                {
                    orow[2*j]   =  irow[2*(M-j)];
                    orow[2*j+1] = -irow[2*(M-j)+1];
                }
            }
        }
    }

    ///@brief Computes forward transform of two interleaved real signals.
    ///       The output can be separated using recover_2x_real_FFT
    ///@param out output array, must not be equal to in
    ///@param out_stride stride, in complex elements, of the output array
    ///@param in1 1st input array
    ///@param in2 2nd input array
    ///@param in_stride1 stride, in scalar elements, of the 1st input array
    ///@param in_stride2 stride, in scalar elements, of the 2nd input array
    void forward_2x(std::complex<T> *out, int out_stride, const T *in1, int in_stride1, const T *in2, int in_stride2)
    {
        scramble_row_x2(out, out_stride, in1, in_stride1, in2, in_stride2, rows());
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

    int vert_cols() const
    {
        return impl::convenient_col_num<float>(cols()/2+1);
    }

    void scramble_row_fwd(std::complex<T> *out, int out_stride, const T *in, int in_stride, int rows)
    {
        if (rows == 1)
        {
            scramble(out, in, cols());
            horz.template transform_no_scramble<false>(out);
        }
        else if (rows == 2)
        {
            const T *in1 = in;
            const T *in2 = in + in_stride;
            scramble((T*)out,   in1, cols(), 2);
            scramble((T*)out+1, in2, cols(), 2);
            horz.template transform_no_scramble<false>(out);
            separate_2x_real_FFT(out, out+out_stride, out, cols());
        }
        else
        {
            scramble_row_fwd(out,             2*out_stride, in,                    in_stride, rows/2);
            scramble_row_fwd(out+out_stride,  2*out_stride, in+(rows/2)*in_stride, in_stride, rows/2);
        }
    }

    void scramble_row_x2(std::complex<T> *out, int out_stride, const T *in1, int in_stride1, const T *in2, int in_stride2, int rows)
    {
        if (rows == 1)
        {
            scramble((T*)out,   in1, cols(), 2);
            scramble((T*)out+1, in2, cols(), 2);
            horz.template transform_no_scramble<false>(out);
        }
        else
        {
            scramble_row_x2(out,             2*out_stride, in1,                     in_stride1, in2,                     in_stride2, rows/2);
            scramble_row_x2(out+out_stride,  2*out_stride, in1+(rows/2)*in_stride1, in_stride1, in2+(rows/2)*in_stride1, in_stride2, rows/2);
        }
    }

    FFT<T>          horz;
    FFTVert<T>      vert;
};

} // genFFT

#endif /* FFTREAL_H */

