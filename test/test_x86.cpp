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
#include <genFFT/x86/x86_features.h>
#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <random>
#include "test_util.h"
#include "fft_test_impl.h"

using genfft::cpu_features;

namespace genfft {
extern cpu_features x86_cpu_features;
} // genfft

static cpu_features feature_params[] = {
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { 1, 0, 0, 0, 0, 0, 0, 0, 0, },
    { 1, 1, 0, 0, 0, 0, 0, 0, 0, },
    { 1, 1, 1, 0, 0, 0, 0, 0, 0, },
    { 1, 1, 1, 1, 0, 0, 0, 0, 0, },
    { 1, 1, 1, 1, 1, 0, 0, 0, 0, },
    { 1, 1, 1, 1, 1, 1, 0, 0, 0, },
    { 1, 1, 1, 1, 1, 1, 0, 0, 1, },
    { 1, 1, 1, 1, 1, 1, 1, 0, 0, },
    { 1, 1, 1, 1, 1, 1, 1, 0, 1, },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, },
};

std::ostream &operator<<(std::ostream &os, cpu_features features)
{
    #define PRINT_FEATURE(name)\
        os << std::left << std::setw(10) << #name ":" << (features.name ? "yes" : "no") << "\n";
    PRINT_FEATURE(SSE)
    PRINT_FEATURE(SSE2)
    PRINT_FEATURE(SSE3)
    PRINT_FEATURE(SSSE3)
    PRINT_FEATURE(SSE41)
    PRINT_FEATURE(SSE42)
    PRINT_FEATURE(AVX)
    PRINT_FEATURE(AVX2)
    PRINT_FEATURE(FMA)
    return os;
}

inline bool FeaturesSupported(cpu_features check, cpu_features present)
{
    const uint8_t *c = (const uint8_t *)&check;
    const uint8_t *p = (const uint8_t *)&present;
    for (size_t i = 0; i < sizeof(cpu_features); i++)
    if ((c[i] & p[i]) != c[i])
        return false;
    return true;
}

TEST(TestArchs, Pow2_float)
{
    cpu_features old_features = genfft::x86_cpu_features;
    for (cpu_features f : feature_params)
    {
        if (!FeaturesSupported(f, old_features))
            continue;
        genfft::x86_cpu_features = f;
        std::cout << "Testing CPU features:\n" << f;
        for (int n = 1; n <= 64; n *= 2)
        {
            std::cout << "Testing FFT size: " << n << "\n";
            TestFFT_Pow2<float>(n);
        }
    }
    genfft::x86_cpu_features = old_features;
}

TEST(TestArchs, Pow2_double)
{
    cpu_features old_features = genfft::x86_cpu_features;
    for (cpu_features f : feature_params)
    {
        if (!FeaturesSupported(f, old_features))
            continue;
        genfft::x86_cpu_features = f;
        std::cout << "Testing CPU features:\n" << f;
        for (int n = 1; n <= 64; n *= 2)
        {
            std::cout << "Testing FFT size: " << n << "\n";
            TestFFT_Pow2<double>(n);
        }
    }
    genfft::x86_cpu_features = old_features;
}

TEST(TestArchs, VertPow2_float)
{
    cpu_features old_features = genfft::x86_cpu_features;
    for (cpu_features f : feature_params)
    {
        if (!FeaturesSupported(f, old_features))
            continue;
        genfft::x86_cpu_features = f;
        std::cout << "Testing CPU features:\n" << f;
        for (int n = 1; n <= 64; n *= 2)
        {
            std::cout << "Testing FFT size: " << n << "\n";
            TestFFTVert_Pow2<float>(n, 63);
        }
    }
    genfft::x86_cpu_features = old_features;
}

TEST(TestArchs, VertPow2_double)
{
    cpu_features old_features = genfft::x86_cpu_features;
    for (cpu_features f : feature_params)
    {
        if (!FeaturesSupported(f, old_features))
            continue;
        genfft::x86_cpu_features = f;
        std::cout << "Testing CPU features:\n" << f;
        for (int n = 1; n <= 64; n *= 2)
        {
            std::cout << "Testing FFT size: " << n << "\n";
            TestFFTVert_Pow2<double>(n, 63);
        }
    }
    genfft::x86_cpu_features = old_features;
}

TEST(TestArchs, DIT_float)
{
    cpu_features old_features = genfft::x86_cpu_features;
    for (cpu_features f : feature_params)
    {
        if (!FeaturesSupported(f, old_features))
            continue;
        genfft::x86_cpu_features = f;
        std::cout << "Testing CPU features:\n" << f;
        for (int n = 1; n <= 64; n *= 2)
        {
            std::cout << "Testing DIT size: " << n << "\n";
            TestDIT_Pow2<float>(n, false);
            std::cout << "Testing DIT (in-place): " << n << "\n";
            TestDIT_Pow2<float>(n, true);
        }
    }
    genfft::x86_cpu_features = old_features;
}

TEST(TestArchs, DIT_double)
{
    cpu_features old_features = genfft::x86_cpu_features;
    for (cpu_features f : feature_params)
    {
        if (!FeaturesSupported(f, old_features))
            continue;
        genfft::x86_cpu_features = f;
        std::cout << "Testing CPU features:\n" << f;
        for (int n = 1; n <= 64; n *= 2)
        {
            std::cout << "Testing DIT size: " << n << "\n";
            TestDIT_Pow2<double>(n, false);
            std::cout << "Testing DIT (in-place): " << n << "\n";
            TestDIT_Pow2<double>(n, true);
        }
    }
    genfft::x86_cpu_features = old_features;
}
