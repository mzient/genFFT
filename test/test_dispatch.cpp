#include <genFFT/fft_dispatch.h>
#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <random>

namespace {

class FFT_test_dispatch : public testing::TestWithParam<int>
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

TEST_P(FFT_test_dispatch, InverseIdentity_float)
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

TEST_P(FFT_test_dispatch, InverseIdentity_double)
{
    int n = GetParam();
    genfft::FFT<double> fft(n);
    std::vector<std::complex<double>> in(n), out(n), invout(n);
    DummyData(in, false);
    fft.transform<false>(out.data(), in.data());
    fft.transform<true>(invout.data(), out.data());
    const double eps = 1e-7;
    const double norm = 1.0/n;
    for (int i = 0; i < n; i++)
    {
        std::complex<double> x = invout[i] * norm;
        ASSERT_NEAR(in[i].real(), x.real(), eps);
        ASSERT_NEAR(in[i].imag(), x.imag(), eps);
    }
}

auto FFT_Sizes = ::testing::Values(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    1<<14, 1<<15, 1<<16, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22);

INSTANTIATE_TEST_CASE_P(
    InverseIdentity_Pow2,
    FFT_test_dispatch,
    FFT_Sizes
);

} // namespace
