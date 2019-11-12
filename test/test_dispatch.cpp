#include <genFFT/fft_dispatch.h>
#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <random>
#include "reference.h"

namespace {

class FFT_test_pow2 : public testing::TestWithParam<int>
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
    1<<14, 1<<15, 1<<16, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22);

INSTANTIATE_TEST_CASE_P(
    FFT_Pow2,
    FFT_test_pow2,
    FFT_Pow2_Sizes
);


} // namespace
