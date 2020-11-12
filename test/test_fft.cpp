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

#include <genFFT/fft.h>
#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <random>

namespace {

class FFT_float_test : public testing::TestWithParam<int>
{
};

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

TEST_P(FFT_float_test, InverseIdentity)
{
    int n = GetParam();
    genfft::FFT<float> fft(n);
    std::vector<std::complex<float>> in(n), out(n), invout(n);
    DummyData(in, false);
    fft.transform<false>(out.data(), in.data());
    fft.transform<true>(invout.data(), out.data());
    const float eps = 1e-4f;
    const float norm = 1.0f/n;
    for (int i = 0; i < n; i++)
    {
        std::complex<float> x = invout[i] * norm;
        ASSERT_NEAR(in[i].real(), x.real(), eps);
        ASSERT_NEAR(in[i].imag(), x.imag(), eps);
    }
}

auto FFT_Sizes = ::testing::Values(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    1<<14, 1<<15, 1<<16, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22);

INSTANTIATE_TEST_CASE_P(
    InverseIdentity,
    FFT_float_test,
    FFT_Sizes
);

} // namespace
