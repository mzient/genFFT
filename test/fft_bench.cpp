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

#include <benchmark/benchmark.h>
#include <genFFT/fft_dispatch.h>
#include <random>
#include <complex>
#include <iostream>
#include <chrono>

using perf_timer = std::chrono::high_resolution_clock;

inline double seconds(perf_timer::time_point start, perf_timer::time_point end)
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

static void FFT_1D(benchmark::State &state)
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

    int64_t total = 0;
    int nn = std::max(4, 1024 / n);

    for (auto _ : state)
    {
        for (auto &c : in)
            c = { dist(rng), dist(rng) };
        auto start = perf_timer::now();
        for (int k = 0; k < nn; k++)
        {
            fft.transform<false>(out.data(), in.data());
            fft.transform<true>(iout.data(), out.data());
        }
        auto end = perf_timer::now();
        double t = seconds(start, end);
        total += nn * n;
        state.SetIterationTime(t);
        niter++;
        for (int i = 0; i < n; i++)
            err += std::abs(iout[i]*den - in[i]);
    }
    state.SetItemsProcessed(total);
    std::cerr << "Average error: " << err / niter << "\n";
}

static void FFT_Vert(benchmark::State &state)
{
    std::mt19937_64 rng;
    std::uniform_real_distribution<float> dist(-1, 1);

    int n = state.range(0);
    int c = state.range(1);
    std::vector<std::complex<float>> in(n*c);
    std::vector<std::complex<float>> out(n*c);
    std::vector<std::complex<float>> iout(n*c);

    genfft::FFTVert<float> fft(n);
    float den = 1.0f/n;
    double err = 0;
    int niter = 0;

    for (auto &c : in)
        c = { dist(rng), dist(rng) };
    fft.transform<false>(out.data(), in.data(), c);
    fft.transform<true>(iout.data(), out.data(), c);

    int nn = std::max(1, 1024 / (n*c));

    for (auto _ : state)
    {
        for (auto &c : in)
            c = { dist(rng), dist(rng) };
        auto start = perf_timer::now();
        for (int k = 0; k < nn; k++)
        {
            fft.transform<false>(out.data(), in.data(), c);
            fft.transform<true>(iout.data(), out.data(), c);
        }
        auto end = perf_timer::now();
        state.SetItemsProcessed(nn);
        state.SetIterationTime(seconds(start, end));
        niter++;
        for (int i = 0; i < n*c; i++)
            err += std::abs(iout[i]*den - in[i]);
    }
    std::cerr << "Average error: " << err / niter << "\n";
}


BENCHMARK(FFT_1D)
    ->UseManualTime()->RangeMultiplier(2)->Range(2, 2<<20);

static void FFTVertArgs(benchmark::internal::Benchmark* b) {
  for (int nfft = 1; nfft <= (1<<20); nfft += nfft) {
      for (int ncols = 1; ncols * nfft < (1<<22); ncols += (ncols <= 16 ? 1 : ncols + 3)) {
          b->Args({ nfft, ncols });
      }
  }
}

BENCHMARK(FFT_Vert)
    ->UseManualTime()->Apply(FFTVertArgs);


BENCHMARK_MAIN();
