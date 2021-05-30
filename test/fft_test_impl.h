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

// NOTE This file must be included after some genfft backend

#ifndef FFT_TEST_IMPL_H
#define FFT_TEST_IMPL_H

#include "test_util.h"
#include "fft_ref_impl.h"

template <typename T>
void TestFFT_Pow2(int n)
{
    genfft::FFT<T> fft(n);
    std::vector<std::complex<T>> in(n), out(n), invout(n);
    std::vector<std::complex<T>> ref_out(n), ref_inv(n);
    DummyData(in, false);
    fft.template transform<false>(out.data(), in.data());
    fft.template transform<true>(invout.data(), out.data());
    reference_impl::FFT_pow2(ref_out.data(), in.data(), n, false);
    reference_impl::FFT_pow2(ref_inv.data(), ref_out.data(), n, true);
    const T eps = FFT_Eps<T>(n);
    const double norm = 1.0/n;
    for (int i = 0; i < n; i++)
    {
        std::complex<T> inv_x(std::complex<double>(invout[i]) * norm);
        std::complex<T> ref_inv_x(std::complex<double>(ref_inv[i]) * norm);
        ASSERT_NEAR(out[i].real(), ref_out[i].real(), eps) << " i = " << i;
        ASSERT_NEAR(out[i].imag(), ref_out[i].imag(), eps) << " i = " << i;
        ASSERT_NEAR(inv_x.real(), ref_inv_x.real(), eps) << " i = " << i;
        ASSERT_NEAR(inv_x.imag(), ref_inv_x.imag(), eps) << " i = " << i;
        ASSERT_NEAR(inv_x.real(), in[i].real(), eps) << " i = " << i;
    }
}

template <typename T>
void TestDIT_Pow2(int n, bool in_place)
{
    std::vector<T> real_input(n);
    std::vector<std::complex<T>> complex_input(n), out_half(n), out1(n), out2(n), out3(n);
    DummyData(real_input);
    for (int i = 0; i < n; i++)
        complex_input[i] = real_input[i];

    genfft::FFT<T> half_fft(n == 1 ? 1 : n/2);
    genfft::FFT<T> full_fft(n);
    half_fft.template transform<false>(out1.data(), (std::complex<T>*)real_input.data());
    full_fft.template transform<false>(out2.data(), complex_input.data());
    genfft::DIT<T> dit(n);
    std::complex<T> *out_ptr = in_place ? out1.data() : out3.data();
    dit.apply(out_ptr, out1.data(), false);
    const T eps = FFT_Eps<T>(n);
    for (int i = 0; i < n; i++)
    {
        EXPECT_NEAR(out_ptr[i].real(), out2[i].real(), eps) << " @ " << i;
        EXPECT_NEAR(out_ptr[i].imag(), out2[i].imag(), eps) << " @ " << i;
    }
}

template <typename T>
void TestRealFFT_Pow2(int n, bool half)
{
    genfft::RealFFT<T> fft(n);
    std::complex<T> fill = { 43, 21 };
    std::vector<std::complex<T>> out(n, fill), ref_out(n);
    std::vector<T> in(n);
    DummyData(in);
    fft.forward(out.data(), in.data(), half);
    reference_impl::FFT_pow2(ref_out.data(), in.data(), n, false);
    const T eps = FFT_Eps<T>(n);
    int limit = half ? n/2 + 1 : n;
    int i;
    for (i = 0; i < limit; i++)
    {
        ASSERT_NEAR(out[i].real(), ref_out[i].real(), eps) << " i = " << i;
        ASSERT_NEAR(out[i].imag(), ref_out[i].imag(), eps) << " i = " << i;
    }
    for (; i < n; i++)
    {
        ASSERT_EQ(out[i], fill) << "Corruption detected @ index " << i;
    }
}

template <typename T>
void TestFFTVert_Pow2(int n, int cols)
{
    genfft::FFTVert<T> fft(n);
    std::vector<std::complex<T>> in(n*cols), out(n*cols), invout(n*cols);
    std::vector<std::complex<T>> ref_out(n*cols), ref_inv(n*cols);
    DummyData(in, false);
    fft.template transform<false>(out.data(), cols, in.data(), cols, cols);
    fft.template transform<true>(invout.data(), cols, out.data(), cols, cols);
    reference_impl::FFT_pow2_vert(ref_out.data(), in.data(), n, cols, false);
    reference_impl::FFT_pow2_vert(ref_inv.data(), ref_out.data(), n, cols, true);
    const T eps = FFT_Eps<T>(n);
    const double norm = 1.0/n;
    for (int i = 0; i < n; i++)
    {
        std::complex<T> inv_x(std::complex<double>(invout[i]) * norm);
        std::complex<T> ref_inv_x(std::complex<double>(ref_inv[i]) * norm);
        ASSERT_NEAR(out[i].real(), ref_out[i].real(), eps) << " i = " << i;
        ASSERT_NEAR(out[i].imag(), ref_out[i].imag(), eps) << " i = " << i;
        ASSERT_NEAR(inv_x.real(), ref_inv_x.real(), eps) << " i = " << i;
        ASSERT_NEAR(inv_x.imag(), ref_inv_x.imag(), eps) << " i = " << i;
        ASSERT_NEAR(inv_x.real(), in[i].real(), eps) << " i = " << i;
    }
}

#endif // FFT_TEST_IMPL_H
