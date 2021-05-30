/*
Copyright 2017-2021 Michal Zientkiewicz

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

#ifndef FFT_X86_UTILS
#define FFT_X86_UTILS

////////////////////////////////////////////////////////////////////

#ifdef GENFFT_USE_AVX
typedef __m256 float8;
inline float8 load(const float *addr)
{
    return _mm256_loadu_ps(addr);
}
inline void store(float *addr, float8 v)
{
    _mm256_storeu_ps(addr, v);
}
inline __m256 flip_even(__m256 a)
{
    const __m256 signmask = _mm256_castsi256_ps(_mm256_set_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000));
    return _mm256_xor_ps(a, signmask);
}
inline __m256 flip_odd(__m256 a)
{
    const __m256 signmask = _mm256_castsi256_ps(_mm256_set_epi32(0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0));
    return _mm256_xor_ps(a, signmask);
}
#else
struct alignas(__m128) float8 { __m128 lo, hi; };
inline float8 load(const float *addr)
{
    return { _mm_loadu_ps(addr), _mm_loadu_ps(addr+4) };
}
inline void store(float *addr, float8 v)
{
    _mm_storeu_ps(addr,     v.lo);
    _mm_storeu_ps(addr + 4, v.hi);
}

#endif

inline __m128 flip_even(__m128 a)
{
    const __m128 signmask = _mm_castsi128_ps(_mm_set_epi32(0, 0x80000000, 0, 0x80000000));
    return _mm_xor_ps(a, signmask);
}

inline __m128 flip_odd(__m128 x)
{
    const __m128 signmask = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0, 0x80000000, 0));
    return _mm_xor_ps(x, signmask);
}

#ifdef GENFFT_USE_SSE3
inline __m128 addsub(__m128 a, __m128 b)
{
    return _mm_addsub_ps(a, b);
}
inline __m128 subadd(__m128 a, __m128 b)
{
    return _mm_addsub_ps(a, -b);
}
#else
inline __m128 addsub(__m128 a, __m128 b)
{
    return _mm_add_ps(a, flip_even(b));
}
inline __m128 subadd(__m128 a, __m128 b)
{
    return _mm_sub_ps(a, flip_even(b));
}
#endif

#ifdef GENFFT_USE_AVX
template <uint8_t mask>
inline __m128 permute(__m128 x)
{
    return _mm_permute_ps(x, mask);
}
#else
template <uint8_t mask>
inline __m128 permute(__m128 x)
{
    return _mm_shuffle_ps(x, x, mask);
}
#endif

////////////////////////////////////////////////////////////////////


#ifdef GENFFT_USE_AVX
typedef __m256d double4;
inline double4 load(const double *addr)
{
    return _mm256_loadu_pd(addr);
}
inline void store(double *addr, double4 v)
{
    _mm256_storeu_pd(addr, v);
}
inline __m256d flip_even(__m256d a)
{
    const __m256d signmask = _mm256_castsi256_pd(_mm256_set_epi32(0, 0, 0x80000000, 0, 0, 0, 0x80000000, 0));
    return _mm256_xor_pd(a, signmask);
}
inline __m256d flip_odd(__m256d a)
{
    const __m256d signmask = _mm256_castsi256_pd(_mm256_set_epi32(0x80000000, 0, 0, 0, 0x80000000, 0, 0, 0));
    return _mm256_xor_pd(a, signmask);
}

#define _MM_SHUFFLE4x2(d, c, b, a) ((d<<3)|(c<<2)|(b<<1)|a)

#else
struct alignas(__m128d) double4 { __m128d lo, hi; };
inline double4 load(const double *addr)
{
    return { _mm_loadu_pd(addr), _mm_loadu_pd(addr+4) };
}
inline void store(double *addr, double4 v)
{
    _mm_storeu_pd(addr,     v.lo);
    _mm_storeu_pd(addr + 4, v.hi);
}

#endif

inline __m128d flip_even(__m128d a)
{
    const __m128d signmask = _mm_castsi128_pd(_mm_set_epi64x(0, 1ull<<63));
    return _mm_xor_pd(a, signmask);
}

inline __m128d flip_odd(__m128d x)
{
    const __m128d signmask = _mm_castsi128_pd(_mm_set_epi64x(1ull<<63, 0));
    return _mm_xor_pd(x, signmask);
}

#ifdef GENFFT_USE_SSE3
inline __m128d addsub(__m128d a, __m128d b)
{
    return _mm_addsub_pd(a, b);
}
inline __m128d subadd(__m128d a, __m128d b)
{
    return _mm_addsub_pd(a, -b);
}
#else
inline __m128d addsub(__m128d a, __m128d b)
{
    return _mm_add_pd(a, flip_even(b));
}
inline __m128d subadd(__m128d a, __m128d b)
{
    return _mm_sub_pd(a, flip_even(b));
}
#endif

#ifdef GENFFT_USE_AVX
template <uint8_t mask>
inline __m128d permute(__m128d x)
{
    return _mm_permute_pd(x, mask);
}
#else
template <uint8_t mask>
inline __m128d permute(__m128d x)
{
    return _mm_shuffle_pd(x, x, mask);
}
#endif

#endif // GENFFT_X86_UTILS
