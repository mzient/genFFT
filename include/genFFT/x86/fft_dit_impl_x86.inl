/*
Copyright 2020-2021 Michal Zientkiewicz

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

#include "../generic/fft_dit_impl_generic.inl"
#include "fft_x86_utils.h"

template <>
inline void adjust_DIT_impl<float>(std::complex<float> *F, const std::complex<float> *Z, int N, bool half, const float *twiddle) noexcept
{
    auto zeroval = Z[0].real() + Z[0].imag();
    if (N == 1)
    {
        F[0] = zeroval;
        return;
    }
    auto centerval = Z[0].real() - Z[0].imag();
    if (N == 2)
    {
        F[0] = zeroval;
        F[1] = centerval;
        return;
    }

    int halfN = N >> 1;
    F[halfN] = Z[0];
    auto quarterval = conj(Z[halfN/2]);
    int i = 0, j = halfN;
#ifdef GENFFT_USE_AVX2
    __m256 halfmul8 = _mm256_set1_ps(0.5f);
    for (; i+4 <= halfN/2; i+=4, j-=4)
    {
        __m256 Ai = _mm256_loadu_ps((const float *)&Z[i]);
        __m256 Bj = _mm256_loadu_ps((const float *)&Z[j-3]);
        Bj = flip_odd(Bj);
        __m256 Bi = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(Bj), _MM_SHUFFLE(0, 1, 2, 3)));
        __m256 O = _mm256_mul_ps(_mm256_sub_ps(Ai, Bi), halfmul8);
        __m256 T = _mm256_load_ps(&twiddle[2*i]);
        __m256 Ti = _mm256_movehdup_ps(T);
        __m256 Tr = _mm256_moveldup_ps(T);
        __m256 OxTi = _mm256_mul_ps(O, Ti);
        OxTi = _mm256_permute_ps(OxTi, _MM_SHUFFLE(2, 3, 0, 1));
#ifdef GENFFT_USE_FMA
        __m256 OxT = _mm256_fmaddsub_ps(O, Tr, OxTi);
#else
        __m256 OxTr = _mm256_mul_ps(O, Tr);
        __m256 OxT = _mm256_addsub_ps(OxTr, OxTi);
#endif
        __m256 E = _mm256_add_ps(O, Bi);
        __m256 Zi = _mm256_sub_ps(E, OxT);
        __m256 Zj = flip_odd(_mm256_add_ps(E, OxT));
        _mm256_storeu_ps((float*)&F[i], Zi);
        Zj = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(Zj), _MM_SHUFFLE(0, 1, 2, 3)));
        _mm256_storeu_ps((float*)&F[j-3], Zj);
    }
#endif
#ifdef GENFFT_USE_SSE
    __m128 halfmul4 = _mm_set1_ps(0.5f);
    for (; i+2 <= halfN/2; i+=2, j-=2)
    {
        __m128 Ai = _mm_loadu_ps((const float *)&Z[i]);
        __m128 Bj = _mm_loadu_ps((const float *)&Z[j-1]);
        __m128 Bi = _mm_shuffle_ps(Bj, Bj, _MM_SHUFFLE(1, 0, 3, 2));
        __m128 O = _mm_mul_ps(addsub(Ai, Bi), halfmul4);
        __m128 T = _mm_load_ps(&twiddle[2*i]);

        __m128 E = subadd(O, Bi);
        __m128 Tr = _mm_shuffle_ps(T, T, _MM_SHUFFLE(2, 2, 0, 0));
        __m128 Ti = _mm_shuffle_ps(T, T, _MM_SHUFFLE(3, 3, 1, 1));
        __m128 OxTi = _mm_mul_ps(O, Ti);
        OxTi = _mm_shuffle_ps(OxTi, OxTi, _MM_SHUFFLE(2, 3, 0, 1));
#ifdef GENFFT_USE_FMA
        __m128 OxT = _mm_fmaddsub_ps(O, Tr, OxTi);
#else
        __m128 OxT = addsub(_mm_mul_ps(O, Tr), OxTi);
#endif
        __m128 Zi = _mm_sub_ps(E, OxT);
        __m128 Zj = flip_odd(_mm_add_ps(E, OxT));
        _mm_storeu_ps((float*)&F[i], Zi);
        _mm_storeu_ps((float*)&F[j-1], _mm_shuffle_ps(Zj, Zj, _MM_SHUFFLE(1, 0, 3, 2)));
    }
#endif
    for (; i < halfN/2; i++, j--)
    {
        std::complex<float> Ei = (Z[i] + conj(Z[j])) * 0.5f;
        std::complex<float> Oi = (Z[i] - conj(Z[j])) * 0.5f;
        std::complex<float> ti = { twiddle[2*i], twiddle[2*i+1] };
        F[i] = Ei - ti*Oi;
        F[j] = conj(Ei + ti*Oi);
    }
    F[0] = zeroval;
    F[halfN/2] = quarterval;
    F[halfN] = centerval;
    if (!half)
        for (int i = halfN+1; i < N; i++)
            F[i] = conj(F[N - i]);
}

template <>
inline void adjust_DIT_impl<double>(std::complex<double> *F, const std::complex<double> *Z, int N, bool half, const double *twiddle) noexcept
{
    auto zeroval = Z[0].real() + Z[0].imag();
    if (N == 1)
    {
        F[0] = zeroval;
        return;
    }
    auto centerval = Z[0].real() - Z[0].imag();
    if (N == 2)
    {
        F[0] = zeroval;
        F[1] = centerval;
        return;
    }

    int halfN = N >> 1;
    F[halfN] = Z[0];
    auto quarterval = conj(Z[halfN/2]);
    int i = 0, j = halfN;
#ifdef GENFFT_USE_SSE2
    __m128d halfmul2 = _mm_set1_pd(0.5);
    for (; i+1 <= halfN/2; i++, j--)
    {
        __m128d Ai = _mm_loadu_pd((const double *)&Z[i]);
        __m128d Bi = _mm_loadu_pd((const double *)&Z[j]);
        __m128d O = _mm_mul_pd(addsub(Ai, Bi), halfmul2);
        __m128d Operm = _mm_shuffle_pd(O, O, _MM_SHUFFLE2(0, 1));
        __m128d T = _mm_load_pd(&twiddle[2*i]);
        __m128d E = subadd(O, Bi);  //_mm_mul_pd(_mm_addsub_pd(Ai, -Bi), halfmul2);
        __m128d Tr = _mm_shuffle_pd(T, T, _MM_SHUFFLE2(0, 0));
        __m128d Ti = _mm_shuffle_pd(T, T, _MM_SHUFFLE2(1, 1));
        __m128d OxTi = _mm_mul_pd(Operm, Ti);
        __m128d OxT = addsub(_mm_mul_pd(O, Tr), OxTi);
        __m128d Zi = _mm_sub_pd(E, OxT);
        __m128d Zj = flip_odd(_mm_add_pd(E, OxT));
        _mm_storeu_pd((double*)&F[i], Zi);
        _mm_storeu_pd((double*)&F[j], Zj);
    }
#endif
    for (; i < halfN/2; i++, j--)
    {
        std::complex<double> Ei = (Z[i] + conj(Z[j])) * 0.5;
        std::complex<double> Oi = (Z[i] - conj(Z[j])) * 0.5;
        int phase = j - halfN/2;
        std::complex<double> ti = { twiddle[2*i], twiddle[2*i+1] };
        F[i] = Ei - ti*Oi;
        F[j] = conj(Ei + ti*Oi);
    }
    F[halfN/2] = quarterval;
    F[0] = zeroval;
    F[halfN] = centerval;
    if (!half)
        for (int i = halfN+1; i < N; i++)
            F[i] = conj(F[N - i]);
}
