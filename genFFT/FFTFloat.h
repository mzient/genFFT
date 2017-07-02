/*
Copyright 2017 Michal Zientkiewicz

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

#ifndef GEN_FFT_FLOAT_H
#define GEN_FFT_FLOAT_H

#ifdef __SSE__
// at least SSE is required
#include <x86intrin.h>
#else
#define GENFFT_NO_FLOAT_INTRIN
#endif

#ifndef GENFFT_NO_FLOAT_INTRIN

namespace genfft {

///@brief Implementation details
namespace impl {

#ifdef __AVX__
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

#ifdef __SSE3__
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

#ifdef __AVX__
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

// Single row FFT for single precision floating point numbers

template <int N, bool GenericCase = (N>8)>
struct FFTFloat : FFTGeneric<N, float>
{
};

template <int N>
struct FFTFloat<N, true>
{
    FFTFloat<N/2> next;
    template <bool inv>
    void transform(float *data)
    {
        next.template transform<inv>(data);
        next.template transform<inv>(data+N);

#ifdef __AVX__
        for (unsigned i=0; i<N; i+=8)
        {
            __m256 W = _mm256_load_ps(twiddle.t + i);
            __m256 E = _mm256_loadu_ps(data+i);
            __m256 O = _mm256_loadu_ps(data+i+N);
            __m256 Wr = _mm256_moveldup_ps(W);
            __m256 Wi = _mm256_movehdup_ps(W);

            __m256 OxWi = _mm256_mul_ps(O, Wi);
            __m256 OxWiperm = _mm256_permute_ps(OxWi, _MM_SHUFFLE(2, 3, 0, 1));
#ifdef __FMA__
            __m256 OxW = inv ? _mm256_fmsubadd_ps(O, Wr, OxWiperm)
                             : _mm256_fmaddsub_ps(O, Wr, OxWiperm);
#else
            __m256 OxWr = _mm256_mul_ps(O, Wr);
            __m256 OxW  = _mm256_addsub_ps(OxWr, inv ? -OxWiperm : OxWiperm);
#endif
            __m256 lo = _mm256_add_ps(E, OxW);
            __m256 hi = _mm256_sub_ps(E, OxW);

            _mm256_storeu_ps(data + i,     lo);
            _mm256_storeu_ps(data + i + N, hi);
        }
#elif defined __SSE__
        for (unsigned i=0; i<N; i+=4)
        {
            __m128 W = _mm_load_ps(twiddle.t + i);
            __m128 E = _mm_loadu_ps(data+i);
            __m128 O = _mm_loadu_ps(data+i+N);
#ifdef __SSE3__
            __m128 Wr = _mm_moveldup_ps(W);
            __m128 Wi = _mm_movehdup_ps(W);
#else
            __m128 Wr = _mm_shuffle_ps(W, W, _MM_SHUFFLE(2, 2, 0, 0));
            __m128 Wi = _mm_shuffle_ps(W, W, _MM_SHUFFLE(3, 3, 1, 1));
#endif

            __m128 OxWi = _mm_mul_ps(O, Wi);
            __m128 OxWr = _mm_mul_ps(O, Wr);
            __m128 Wxiperm = permute<_MM_SHUFFLE(2, 3, 0, 1)>(OxWi);
            __m128 OxW  = inv ? subadd(OxWr, Wxiperm) : addsub(OxWr, Wxiperm);
            __m128 lo = _mm_add_ps(E, OxW);
            __m128 hi = _mm_sub_ps(E, OxW);

            _mm_storeu_ps(data + i,     lo);
            _mm_storeu_ps(data + i + N, hi);
        }
#endif

    }

    const Twiddle<N, float> twiddle;
};

template <>
struct FFTFloat<4>
{
    template <bool inv>
    float8 transform_ex(float *data)
    {
        __m128 xlo   = _mm_loadu_ps(data);
        __m128 xhi   = _mm_loadu_ps(data + 4);

        __m128 xa = _mm_shuffle_ps(xlo, xhi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 xb = _mm_shuffle_ps(xlo, xhi, _MM_SHUFFLE(3, 2, 3, 2));

        __m128 ya = _mm_add_ps(xa, xb);
        __m128 yb = _mm_sub_ps(xa, xb);

        // flip sign in one lane
        __m128 mask = _mm_castsi128_ps(inv ? _mm_set_epi32(0x80000000u, 0, 0, 0) : _mm_set_epi32(0, 0x80000000u, 0, 0));
        yb = _mm_xor_ps(yb, mask);

        __m128 za = _mm_shuffle_ps(ya, yb, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 zb = _mm_shuffle_ps(ya, yb, _MM_SHUFFLE(2, 3, 3, 2));

        ya = _mm_add_ps(za, zb);
        yb = _mm_sub_ps(za, zb);
#ifdef __AVX__
        return _mm256_insertf128_ps(_mm256_castps128_ps256(ya), yb, 1);
#else
        return { ya, yb };
#endif
    }

    template <bool inv>
    void transform(float *data)
    {
        store(data, transform_ex<inv>(data));
    }

};

#ifdef __AVX__
template <>
struct FFTFloat<8>
{
    static const int N = 8;
    FFTFloat<4> next;

    template <bool inv>
    void transform(float *data)
    {
        static const Twiddle<N, float> twiddle;

        __m256 E = next.transform_ex<inv>(data);
        __m256 O = next.transform_ex<inv>(data+N);

        __m256 W = _mm256_load_ps(twiddle.t);
        __m256 Wr = _mm256_moveldup_ps(W);
        __m256 Wi = _mm256_movehdup_ps(W);

        __m256 OxWi = _mm256_mul_ps(O, Wi);
        __m256 OxWiperm = _mm256_permute_ps(OxWi, _MM_SHUFFLE(2, 3, 0, 1));
#ifdef __FMA__
        __m256 OxW = inv ? _mm256_fmsubadd_ps(O, Wr, OxWiperm)
                         : _mm256_fmaddsub_ps(O, Wr, OxWiperm);

#else
        __m256 OxWr = _mm256_mul_ps(O, Wr);
        __m256 OxW  = _mm256_addsub_ps(OxWr, inv ? -OxWiperm : OxWiperm);
#endif
        __m256 lo = _mm256_add_ps(E, OxW);
        __m256 hi = _mm256_sub_ps(E, OxW);

        _mm256_storeu_ps(data,     lo);
        _mm256_storeu_ps(data + N, hi);
    }
};
#endif

template <int N>
struct FFTImplSelector<N, float>
{
    typedef impl::FFTFloat<N> type;
};

// Vertical multi-column FFT for singgle precision floating point values

template <int N, bool GenericCase=(N>4)>
struct FFTVertFloat : FFTVertGeneric<N, float> {};

template <int N>
struct FFTVertFloat<N, true>
{
    FFTVertFloat<N/2> next;

    template <bool inv>
    void transform(float *data, int stride, int cols)
    {
        int half = N/2 * stride;
        next.transform<inv>(data,      stride, cols);
        next.transform<inv>(data+half, stride, cols);

        for (unsigned i=0; i<N/2; i++)
        {
            float *even = data + i*stride;
            float *odd  = even + half;

            int j=0;
#ifdef __AVX__
            __m256 Wr = _mm256_broadcast_ss(&twiddle.t[2*i]);
            __m256 Wi = _mm256_broadcast_ss(&twiddle.t[2*i+1]);

            for (; j+8<=2*cols; j+=8)
            {
                __m256 E = _mm256_loadu_ps(even+j);
                __m256 O = _mm256_loadu_ps(odd+j);

                __m256 OxWi = _mm256_mul_ps(O, Wi);
                __m256 OxWiperm = _mm256_permute_ps(OxWi, _MM_SHUFFLE(2, 3, 0, 1));
#ifdef __FMA__
                __m256 OxW  = inv ? _mm256_fmsubadd_ps(O, Wr, OxWiperm)
                                  : _mm256_fmaddsub_ps(O, Wr, OxWiperm);
#else
                __m256 OxWr = _mm256_mul_ps(O, Wr);
                __m256 OxW  = _mm256_addsub_ps(OxWr, inv ? -OxWiperm : OxWiperm);
#endif
                __m256 lo = _mm256_add_ps(E, OxW);
                __m256 hi = _mm256_sub_ps(E, OxW);

                _mm256_storeu_ps(even+j, lo);
                _mm256_storeu_ps(odd+j,  hi);
            }

            __m128 Wr128 = _mm256_castps256_ps128(Wr);
            __m128 Wi128 = _mm256_castps256_ps128(Wi);
#else
            __m128 Wr128 = _mm_set1_ps(twiddle[2*i]);
            __m128 Wi128 = _mm_set1_ps(twiddle[2*i+1]);
#endif

            for (; j+4<=2*cols; j+=4)
            {
                __m128 E = _mm_loadu_ps(even+j);
                __m128 O = _mm_loadu_ps(odd+j);

                __m128 OxWi = _mm_mul_ps(O, Wi128);
                __m128 Wxiperm = permute<_MM_SHUFFLE(2, 3, 0, 1)>(OxWi);
#ifdef __FMA__
                __m128 OxW  = inv ? _mm_fmsubadd_ps(O, Wr128, Wxiperm)
                                  : _mm_fmaddsub_ps(O, Wr128, Wxiperm);
#else
                __m128 OxWr = _mm_mul_ps(O, Wr128);
                __m128 OxW  = inv ? subadd(OxWr, Wxiperm) : addsub(OxWr, Wxiperm);
#endif
                __m128 lo = _mm_add_ps(E, OxW);
                __m128 hi = _mm_sub_ps(E, OxW);

                _mm_storeu_ps(even+j, lo);
                _mm_storeu_ps(odd+j,  hi);
            }

            float wr = twiddle[2*i];
            float wi = twiddle[2*i+1];
            for (; j<2*cols; j+=2)
            {
                float tempr = inv ? odd[j]*wr + odd[j+1]*wi : odd[j]*wr - odd[j+1]*wi;
                float tempi = inv ? odd[j+1]*wr -odd[j]*wi : odd[j]*wi + odd[j+1]*wr;
                odd[j]    = even[j]-tempr;
                odd[j+1]  = even[j+1]-tempi;
                even[j]   += tempr;
                even[j+1] += tempi;
            }
        }
    }

    const Twiddle<N, float> twiddle;
};


template <>
struct FFTVertFloat<4>
{
    template <bool inv>
    void transform(float *data, int stride, int cols)
    {
        float *row0 = data;
        float *row1 = row0+stride;
        float *row2 = row1+stride;
        float *row3 = row2+stride;

        int j = 0;
#ifdef __AVX__
        for (; j+8<=cols*2; j+=8)
        {
            __m256 x0 = _mm256_loadu_ps(row0+j);
            __m256 x1 = _mm256_loadu_ps(row1+j);
            __m256 x2 = _mm256_loadu_ps(row2+j);
            __m256 x3 = _mm256_loadu_ps(row3+j);

            __m256 y0 = _mm256_add_ps(x0, x1);
            __m256 y1 = _mm256_sub_ps(x0, x1);
            __m256 y2 = _mm256_add_ps(x2, x3);

            enum {
                mask1 = inv ? 0xaa : 0x55,
                mask2 = inv ? 0x55 : 0xaa
            };
            __m256 y3 = _mm256_permute_ps(_mm256_sub_ps(_mm256_blend_ps(x2, x3, mask1), _mm256_blend_ps(x2, x3, mask2)), _MM_SHUFFLE(2,3,0,1));

            __m256 z0 = _mm256_add_ps(y0, y2);
            __m256 z1 = _mm256_add_ps(y1, y3);
            __m256 z2 = _mm256_sub_ps(y0, y2);
            __m256 z3 = _mm256_sub_ps(y1, y3);

            _mm256_storeu_ps(row0+j, z0);
            _mm256_storeu_ps(row1+j, z1);
            _mm256_storeu_ps(row2+j, z2);
            _mm256_storeu_ps(row3+j, z3);
        }
#endif

        for (; j+4<=cols*2; j+=4)
        {
            __m128 x0 = _mm_loadu_ps(row0+j);
            __m128 x1 = _mm_loadu_ps(row1+j);
            __m128 x2 = _mm_loadu_ps(row2+j);
            __m128 x3 = _mm_loadu_ps(row3+j);

            __m128 y0 = _mm_add_ps(x0, x1);
            __m128 y1 = _mm_sub_ps(x0, x1);
            __m128 y2 = _mm_add_ps(x2, x3);
#ifdef __SSE4_1__
            enum {
                mask1 = inv ? 0xa : 0x5,
                mask2 = inv ? 0x5 : 0xa
            };
            __m128 y3 = permute<_MM_SHUFFLE(2,3,0,1)>(_mm_sub_ps(_mm_blend_ps(x2, x3, mask1), _mm_blend_ps(x2, x3, mask2)));
#else
            __m128 y3 = permute<_MM_SHUFFLE(2,3,0,1)>(flip_even(inv ? _mm_sub_ps(x3, x2) : _mm_sub_ps(x2, x3)));
#endif

            __m128 z0 = _mm_add_ps(y0, y2);
            __m128 z1 = _mm_add_ps(y1, y3);
            __m128 z2 = _mm_sub_ps(y0, y2);
            __m128 z3 = _mm_sub_ps(y1, y3);

            _mm_storeu_ps(row0+j, z0);
            _mm_storeu_ps(row1+j, z1);
            _mm_storeu_ps(row2+j, z2);
            _mm_storeu_ps(row3+j, z3);
        }

        // tail
        FFTVertGeneric<4, float>().transform<inv>(data + j, stride, cols - j/2);
    }

};

template <>
struct FFTVertFloat<2>
{
    template <bool inv_unused>
    void transform(float *data, int stride, int cols)
    {
        float *row0 = data;
        float *row1 = row0+stride;

        int j = 0;
#ifdef __AVX__
        for (; j+8<=cols*2; j+=8)
        {
            __m256 x0 = _mm256_loadu_ps(row0+j);
            __m256 x1 = _mm256_loadu_ps(row1+j);
            _mm256_storeu_ps(row0+j, _mm256_add_ps(x0, x1));
            _mm256_storeu_ps(row1+j, _mm256_sub_ps(x0, x1));
        }
#endif
#ifdef __SSE__
        for (; j+4<=cols*2; j+=4)
        {
            __m128 x0 = _mm_loadu_ps(row0+j);
            __m128 x1 = _mm_loadu_ps(row1+j);
            _mm_storeu_ps(row0+j, _mm_add_ps(x0, x1));
            _mm_storeu_ps(row1+j, _mm_sub_ps(x0, x1));
        }
#endif
        for (; j<cols*2; j++)
        {
            float x0 = row0[j];
            float x1 = row1[j];
            row0[j] = x0 + x1;
            row1[j] = x0 - x1;
        }
    }

};

template <int N>
struct FFTVertImplSelector<N, float>
{
    typedef impl::FFTVertFloat<N> type;
};

} // impl
} // genfft

#endif

#endif /* GEN_FFT_FLOAT_H */
