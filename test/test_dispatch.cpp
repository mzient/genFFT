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

#include <genFFT/fft_dispatch.h>
#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <random>
#include "test_util.h"
#include "fft_test_impl.h"

namespace {

///////////////////////////////////////////////////////////////
// 1D FFT test

class FFT_test_pow2 : public testing::TestWithParam<int>
{
};


TEST_P(FFT_test_pow2, Pow2_float)
{
    int n = GetParam();
    TestFFT_Pow2<float>(n);
}

TEST_P(FFT_test_pow2, Pow2_double)
{
    int n = GetParam();
    TestFFT_Pow2<double>(n);
}


auto FFT_Pow2_Sizes = ::testing::Values(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    1<<14, 1<<15, 1<<16, 1<<17, 1<<18, 1<<19, 1<<20);

INSTANTIATE_TEST_CASE_P(
    FFT_Pow2,
    FFT_test_pow2,
    FFT_Pow2_Sizes
);

///////////////////////////////////////////////////////////////
// Vertical FFT test

struct VertFFTParam
{
    int nfft, cols;
};

inline void PrintTo(const VertFFTParam &param, std::ostream *os)
{
    *os << "nfft=" << param.nfft << " cols=" << param.cols;
}

class FFT_vert_test_pow2 : public testing::TestWithParam<VertFFTParam>
{
};

TEST_P(FFT_vert_test_pow2, Pow2_float)
{
    auto param = GetParam();
    TestFFTVert_Pow2<float>(param.nfft, param.cols);
}

TEST_P(FFT_vert_test_pow2, Pow2_double)
{
    auto param = GetParam();
    TestFFTVert_Pow2<double>(param.nfft, param.cols);
}

std::vector<VertFFTParam> FFT_Vert_Pow2_Sizes = {
    { 1, 1 },
    { 1, 40 },
    { 2, 1 },
    { 2, 7 },
    { 2, 999 },
    { 4, 1 },
    { 4, 7 },
    { 4, 999 },
    { 8, 1 },
    { 8, 3 },
    { 8, 7 },
    { 8, 33 },
    { 8, 47 },
    { 8, 1023 },
    { 16, 1 },
    { 16, 47 },
    { 16, 999 },
    { 256, 1 },
    { 256, 1023 },
    { 1024, 1 },
    { 1024, 767 },
    { 65536, 1 },
    { 65536, 31 }
};

INSTANTIATE_TEST_CASE_P(
    FFT_Vert_Pow2,
    FFT_vert_test_pow2,
    ::testing::ValuesIn(FFT_Vert_Pow2_Sizes)
);


} // namespace
