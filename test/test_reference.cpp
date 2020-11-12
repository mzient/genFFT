/*
Copyright 2019 Michal Zientkiewicz

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

#include "test_util.h"
#include <gtest/gtest.h>
#include <random>

TEST(ReferenceImplTest, FFTvsDFT)
{
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> dist(-1, 1);

    std::vector<std::complex<double>> in(1024);
    std::vector<std::complex<double>> out_fft(1024);
    std::vector<std::complex<double>> out_dft(1024);

    for (auto &c : in)
        c = { dist(rng), dist(rng) };

    for (int n = 2; n < 1024; n += n)
    {
        for (bool dir : { false, true } )
        {
            for (int i = 0; i < n; i++)
            {
                out_fft[i] = std::nan("1");
                out_dft[i] = std::nan("2");
            }

            reference_impl::DFT(out_dft.data(), in.data(), n, dir);
            reference_impl::FFT_pow2(out_fft.data(), in.data(), n, dir);

            double eps = 1e-10 + 1e-11 * n;

            for (int i = 0; i < n; i++)
            {
                EXPECT_NEAR(out_fft[i].real(), out_dft[i].real(), eps);
                EXPECT_NEAR(out_fft[i].imag(), out_dft[i].imag(), eps);
            }
        }
    }
}
