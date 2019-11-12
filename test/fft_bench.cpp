#include <benchmark/benchmark.h>
#include <genFFT/fft_dispatch.h>
#include <random>
#include <complex>
#include <iostream>

static void FFT_Forward(benchmark::State &state)
{
    std::mt19937_64 rng;
    std::uniform_real_distribution<float> dist(-1, 1);

    int n = state.range(0);
    std::vector<std::complex<float>> in(n);
    std::vector<std::complex<float>> out(n);
    std::vector<std::complex<float>> iout(n);

    genfft::FFT<float> fft(n);
    float den = 1.0f/n;
    double err = 0;
    int niter = 0;

    for (auto &c : in)
        c = { dist(rng), dist(rng) };
    fft.transform<false>(out.data(), in.data());
    fft.transform<true>(iout.data(), out.data());

    for (auto _ : state)
    {
        state.PauseTiming();
        for (auto &c : in)
            c = { dist(rng), dist(rng) };
        state.ResumeTiming();
        for (int k = 0; k < 100; k++) {
            fft.transform<false>(out.data(), in.data());
            fft.transform<true>(iout.data(), out.data());
        }
        state.PauseTiming();
        niter++;
        for (int i = 0; i < n; i++)
            err += std::abs(iout[i]*den - in[i]);
        state.ResumeTiming();
    }
    std::cerr << "Average error: " << err / niter << "\n";
}

BENCHMARK(FFT_Forward)
    ->RangeMultiplier(2)->Range(2, 2<<20);

BENCHMARK_MAIN();
