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
#ifndef GEN_FFT_DOUBLE_H
#define GEN_FFT_DOUBLE_H

#ifdef __SSE2__
// at least SSE2 is required
#include <x86intrin.h>
#else
#define GENFFT_NO_DOUBLE_INTRIN
#endif

#ifndef GENFFT_NO_DOUBLE_INTRIN

///@brief Implementation details
namespace genfft {
namespace impl {

#ifdef __AVX__
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
    const __m256d signmask = _mm256_castsi256_pd(_mm256_set_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000));
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

#ifdef __SSE3__
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

#ifdef __AVX__
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

// Single row FFT for single precision doubleing point numbers

template <int N, bool GenericCase = (N>2)>
struct FFTDouble : FFTGeneric<N, double>
{
};

template <int N>
struct FFTDouble<N, true>
{
    FFTDouble<N/2> next;
    void forward(double *data)
    {
        next.forward(data);
        next.forward(data+N);

#ifdef __AVX__
        for (unsigned i=0; i<N; i+=4)
        {
            __m256d W = _mm256_load_pd(twiddle.t + i);
            __m256d E = _mm256_loadu_pd(data+i);
            __m256d O = _mm256_loadu_pd(data+i+N);
            __m256d Wr = _mm256_permute_pd(W, _MM_SHUFFLE4x2(0, 0, 0, 0));
            __m256d Wi = _mm256_permute_pd(W, _MM_SHUFFLE4x2(1, 1, 1, 1));
            
            __m256d OxWi = _mm256_mul_pd(O, Wi);
            __m256d Wxiperm = _mm256_permute_pd(OxWi, _MM_SHUFFLE4x2(0, 1, 0, 1));
#ifdef __FMA__
            __m256d OxW  = _mm256_fmaddsub_pd(O, Wr, Wxiperm);
#else
            __m256d OxWr = _mm256_mul_pd(O, Wr);
            __m256d OxW  = _mm256_addsub_pd(OxWr, _mm256_permute_pd(OxWi, _MM_SHUFFLE4x2(0, 1, 0, 1)));
#endif
            __m256d lo = _mm256_add_pd(E, OxW);
            __m256d hi = _mm256_sub_pd(E, OxW);

            _mm256_storeu_pd(data + i,     lo);
            _mm256_storeu_pd(data + i + N, hi);
        }
#elif defined __SSE2__
        for (unsigned i=0; i<N; i+=2)
        {
            __m128d W = _mm_load_pd(twiddle.t + i);
            __m128d E = _mm_loadu_pd(data+i);
            __m128d O = _mm_loadu_pd(data+i+N);

            __m128d Wr = _mm_shuffle_pd(W, W, _MM_SHUFFLE2(0, 0));
            __m128d Wi = _mm_shuffle_pd(W, W, _MM_SHUFFLE2(1, 1));

            __m128d OxWi = _mm_mul_pd(O, Wi);
            __m128d OxWr = _mm_mul_pd(O, Wr);
            __m128d Wxiperm = permute<_MM_SHUFFLE2(0, 1)>(OxWi);
            __m128d OxW  = addsub(OxWr, Wxiperm);
            __m128d lo = _mm_add_pd(E, OxW);
            __m128d hi = _mm_sub_pd(E, OxW);

            _mm_storeu_pd(data + i,     lo);
            _mm_storeu_pd(data + i + N, hi);
        }
#endif

    }
    void inverse(double *data)
    {
        next.inverse(data);
        next.inverse(data+N);

#ifdef __AVX__
        for (unsigned i=0; i<N; i+=4)
        {
            __m256d W = _mm256_load_pd(twiddle.t + i);
            __m256d E = _mm256_loadu_pd(data+i);
            __m256d O = _mm256_loadu_pd(data+i+N);
            __m256d Wr = _mm256_permute_pd(W, _MM_SHUFFLE4x2(0, 0, 0, 0));
            __m256d Wi = _mm256_permute_pd(W, _MM_SHUFFLE4x2(1, 1, 1, 1));

            __m256d OxWi = _mm256_mul_pd(O, Wi);
#ifdef __FMA__
            __m256d OxW  = _mm256_fmsubadd_pd(O, Wr, _mm256_permute_pd(OxWi, _MM_SHUFFLE4x2(0, 1, 0, 1)));
#else
            __m256d OxWr = _mm256_mul_pd(O, Wr);
            __m256d OxW  = _mm256_addsub_pd(OxWr, -_mm256_permute_pd(OxWi, _MM_SHUFFLE4x2(0, 1, 0, 1)));
#endif

            __m256d lo = _mm256_add_pd(E, OxW);
            __m256d hi = _mm256_sub_pd(E, OxW);

            _mm256_storeu_pd(data + i,     lo);
            _mm256_storeu_pd(data + i + N, hi);
        }
#elif defined __SSE2__
        for (unsigned i=0; i<N; i+=2)
        {
            __m128d W = _mm_load_pd(twiddle.t + i);
            __m128d E = _mm_loadu_pd(data+i);
            __m128d O = _mm_loadu_pd(data+i+N);

            __m128d Wr = _mm_shuffle_pd(W, W, _MM_SHUFFLE2(0, 0));
            __m128d Wi = _mm_shuffle_pd(W, W, _MM_SHUFFLE2(1, 1));

            __m128d OxWi = _mm_mul_pd(O, Wi);
            __m128d OxWr = _mm_mul_pd(O, Wr);
            __m128d Wxiperm = permute<_MM_SHUFFLE2(0, 1)>(OxWi);
            __m128d OxW  = subadd(OxWr, Wxiperm);
            __m128d lo = _mm_add_pd(E, OxW);
            __m128d hi = _mm_sub_pd(E, OxW);

            _mm_storeu_pd(data + i,     lo);
            _mm_storeu_pd(data + i + N, hi);
        }
#endif
    }


    const Twiddle<N, double> twiddle;
};

template <>
struct FFTDouble<4>
{
    void forward(double *data)
    {
        __m128d x45 = _mm_loadu_pd(data+4);
        __m128d x67 = _mm_loadu_pd(data+6);

        __m128d y45 = _mm_add_pd(x45, x67);

        __m128d y76 = _mm_sub_pd(x45, x67);
        __m128d y67 = permute<_MM_SHUFFLE2(0,1)>(flip_even(y76));
        
        __m128d x01 = _mm_loadu_pd(data);
        __m128d x23 = _mm_loadu_pd(data+2);

        __m128d y01 = _mm_add_pd(x01, x23);
        __m128d y23 = _mm_sub_pd(x01, x23);
#ifdef __AVX__
        __m256d y0123 = _mm256_insertf128_pd(_mm256_castpd128_pd256(y01), y23, 1);
        __m256d y4567 = _mm256_insertf128_pd(_mm256_castpd128_pd256(y45), y67, 1);
        __m256d z0 = _mm256_add_pd(y0123, y4567);
        _mm256_storeu_pd(data,     z0);
        __m256d z1 = _mm256_sub_pd(y0123, y4567);
        _mm256_storeu_pd(data + 4, z1);
#else
        __m128d z0 = _mm_add_pd(y01, y45);
        __m128d z1 = _mm_add_pd(y23, y67);
        __m128d z2 = _mm_sub_pd(y01, y45);
        __m128d z3 = _mm_sub_pd(y23, y67);

        _mm_storeu_pd(data,   z0);
        _mm_storeu_pd(data+2, z1);
        _mm_storeu_pd(data+4, z2);
        _mm_storeu_pd(data+6, z3);
#endif
    }

    void inverse(double *data)
    {
        __m128d x45 = _mm_loadu_pd(data+4);
        __m128d x67 = _mm_loadu_pd(data+6);

        __m128d y45 = _mm_add_pd(x45, x67);

        __m128d y76 = _mm_sub_pd(x45, x67);
        __m128d y67 = flip_even(permute<_MM_SHUFFLE2(0,1)>(y76));

        __m128d x01 = _mm_loadu_pd(data);
        __m128d x23 = _mm_loadu_pd(data+2);

        __m128d y01 = _mm_add_pd(x01, x23);
        __m128d y23 = _mm_sub_pd(x01, x23);

#ifdef __AVX__
        __m256d y0123 = _mm256_insertf128_pd(_mm256_castpd128_pd256(y01), y23, 1);
        __m256d y4567 = _mm256_insertf128_pd(_mm256_castpd128_pd256(y45), y67, 1);
        __m256d z0 = _mm256_add_pd(y0123, y4567);
        _mm256_storeu_pd(data,     z0);
        __m256d z1 = _mm256_sub_pd(y0123, y4567);
        _mm256_storeu_pd(data + 4, z1);
#else
        __m128d z0 = _mm_add_pd(y01, y45);
        __m128d z1 = _mm_add_pd(y23, y67);
        __m128d z2 = _mm_sub_pd(y01, y45);
        __m128d z3 = _mm_sub_pd(y23, y67);

        _mm_storeu_pd(data,   z0);
        _mm_storeu_pd(data+2, z1);
        _mm_storeu_pd(data+4, z2);
        _mm_storeu_pd(data+6, z3);
#endif
    }
};

template <int N>
struct FFTImplSelector<N, double>
{
    typedef impl::FFTDouble<N> type;
};

// Vertical multi-column FFT for singgle precision doubleing point values
template <int N, bool GenericCase=(N>2)>
struct FFTVertDouble : FFTVertGeneric<N, double> {};

template <int N>
struct FFTVertDouble<N, true>
{
    FFTVertDouble<N/2> next;
    void forward(double *data, int stride, int cols)
    {
        int half = N/2 * stride;
        next.forward(data,      stride, cols);
        next.forward(data+half, stride, cols);

        for (unsigned i=0; i<N/2; i++)
        {
            double *even = data + i*stride;
            double *odd  = even + half;

            int j=0;
#ifdef __AVX__
            __m256d Wr = _mm256_broadcast_sd(&twiddle.t[2*i]);
            __m256d Wi = _mm256_broadcast_sd(&twiddle.t[2*i+1]);

            for (; j+4<=2*cols; j+=4)
            {
                __m256d E = _mm256_loadu_pd(even+j);
                __m256d O = _mm256_loadu_pd(odd+j);

                __m256d OxWi = _mm256_mul_pd(O, Wi);
#ifdef __FMA__
                __m256d OxW  = _mm256_fmaddsub_pd(O, Wr, _mm256_permute_pd(OxWi, _MM_SHUFFLE4x2(0, 1, 0, 1)));
#else
                __m256d OxWr = _mm256_mul_pd(O, Wr);
                __m256d OxW  = _mm256_addsub_pd(OxWr, _mm256_permute_pd(OxWi, _MM_SHUFFLE4x2(0, 1, 0, 1)));
#endif
                __m256d lo = _mm256_add_pd(E, OxW);
                __m256d hi = _mm256_sub_pd(E, OxW);

                _mm256_storeu_pd(even+j, lo);
                _mm256_storeu_pd(odd+j,  hi);
            }

            __m128d Wr128 = _mm256_castpd256_pd128(Wr);
            __m128d Wi128 = _mm256_castpd256_pd128(Wi);
#else
            __m128d Wr128 = _mm_set1_pd(twiddle[2*i]);
            __m128d Wi128 = _mm_set1_pd(twiddle[2*i+1]);
#endif

            for (; j+2<=2*cols; j+=2)
            {
                __m128d E = _mm_loadu_pd(even+j);
                __m128d O = _mm_loadu_pd(odd+j);

                __m128d OxWi = _mm_mul_pd(O, Wi128);
                __m128d Wxiperm = permute<_MM_SHUFFLE2(0, 1)>(OxWi);
#ifdef __FMA__
                __m128d OxW  = _mm_fmaddsub_pd(O, Wr128, Wxiperm);
#else
                __m128d OxWr = _mm_mul_pd(O, Wr128);
                __m128d OxW  = addsub(OxWr, Wxiperm);
#endif
                __m128d lo = _mm_add_pd(E, OxW);
                __m128d hi = _mm_sub_pd(E, OxW);

                _mm_storeu_pd(even+j, lo);
                _mm_storeu_pd(odd+j,  hi);
            }

            double wr = twiddle[2*i];
            double wi = twiddle[2*i+1];
            for (; j<2*cols; j+=2)
            {
                double tempr = odd[j]*wr - odd[j+1]*wi;
                double tempi = odd[j]*wi + odd[j+1]*wr;
                odd[j]    = even[j]-tempr;
                odd[j+1]  = even[j+1]-tempi;
                even[j]   += tempr;
                even[j+1] += tempi;
            }
        }
    }
    void inverse(double *data, int stride, int cols)
    {
        int half = N/2 * stride;
        next.inverse(data,      stride, cols);
        next.inverse(data+half, stride, cols);

        for (unsigned i=0; i<N/2; i++)
        {
            double *even = data + i*stride;
            double *odd  = even + half;

            int j=0;
#ifdef __AVX__
            __m256d Wr = _mm256_broadcast_sd(&twiddle.t[2*i]);
            __m256d Wi = _mm256_broadcast_sd(&twiddle.t[2*i+1]);

            for (; j+4<=2*cols; j+=4)
            {
                __m256d E = _mm256_loadu_pd(even+j);
                __m256d O = _mm256_loadu_pd(odd+j);

                __m256d OxWi = _mm256_mul_pd(O, Wi);
#ifdef __FMA__
                __m256d OxW  = _mm256_fmsubadd_pd(O, Wr, _mm256_permute_pd(OxWi, _MM_SHUFFLE4x2(0, 1, 0, 1)));
#else
                __m256d OxWr = _mm256_mul_pd(O, Wr);
                __m256d OxW  = _mm256_addsub_pd(OxWr, -_mm256_permute_pd(OxWi, _MM_SHUFFLE4x2(0, 1, 0, 1)));
#endif

                __m256d lo = _mm256_add_pd(E, OxW);
                __m256d hi = _mm256_sub_pd(E, OxW);
                _mm256_storeu_pd(even+j, lo);
                _mm256_storeu_pd(odd+j,  hi);
            }

            __m128d Wr128 = _mm256_castpd256_pd128(Wr);
            __m128d Wi128 = _mm256_castpd256_pd128(Wi);
#else
            __m128d Wr128 = _mm_set1_pd(twiddle[2*i]);
            __m128d Wi128 = _mm_set1_pd(twiddle[2*i+1]);
#endif

            for (; j+2<=2*cols; j+=2)
            {
                __m128d E = _mm_loadu_pd(even+j);
                __m128d O = _mm_loadu_pd(odd+j);

                __m128d OxWi = _mm_mul_pd(O, Wi128);
                __m128d Wxiperm = permute<_MM_SHUFFLE2(0, 1)>(OxWi);
#ifdef __FMA__
                __m128d OxW  = _mm_fmsubadd_pd(O, Wr128, Wxiperm);
#else
                __m128d OxWr = _mm_mul_pd(O, Wr128);
                __m128d OxW  = subadd(OxWr, Wxiperm);
#endif

                __m128d lo = _mm_add_pd(E, OxW);
                __m128d hi = _mm_sub_pd(E, OxW);
                _mm_storeu_pd(even+j, lo);
                _mm_storeu_pd(odd+j,  hi);
            }

            double wr = twiddle[2*i];
            double wi = twiddle[2*i+1];
            for (; j<2*cols; j+=2)
            {
                double tempr = odd[j]*wr + odd[j+1]*wi;
                double tempi = odd[j+1]*wr - odd[j]*wi;
                odd[j]    = even[j]-tempr;
                odd[j+1]  = even[j+1]-tempi;
                even[j]   += tempr;
                even[j+1] += tempi;
            }

        }
    }


    const Twiddle<N, double> twiddle;
};


template <>
struct FFTVertDouble<4>
{
    void forward(double *data, int stride, int cols)
    {
        double *row0 = data;
        double *row1 = row0+stride;
        double *row2 = row1+stride;
        double *row3 = row2+stride;

        int j = 0;
#ifdef __AVX__
        for (; j+4<=cols*2; j+=4)
        {
            __m256d x0 = _mm256_loadu_pd(row0+j);
            __m256d x1 = _mm256_loadu_pd(row1+j);
            __m256d x2 = _mm256_loadu_pd(row2+j);
            __m256d x3 = _mm256_loadu_pd(row3+j);

            __m256d y0 = _mm256_add_pd(x0, x1);
            __m256d y1 = _mm256_sub_pd(x0, x1);
            __m256d y2 = _mm256_add_pd(x2, x3);
            __m256d x2332 = _mm256_sub_pd(_mm256_blend_pd(x2, x3, 0x5), _mm256_blend_pd(x2, x3, 0xa));
            __m256d y3 = _mm256_permute_pd(x2332, _MM_SHUFFLE4x2(0, 1, 0, 1));

            __m256d z0 = _mm256_add_pd(y0, y2);
            __m256d z1 = _mm256_add_pd(y1, y3);
            __m256d z2 = _mm256_sub_pd(y0, y2);
            __m256d z3 = _mm256_sub_pd(y1, y3);

            _mm256_storeu_pd(row0+j, z0);
            _mm256_storeu_pd(row1+j, z1);
            _mm256_storeu_pd(row2+j, z2);
            _mm256_storeu_pd(row3+j, z3);
        }
#endif

        for (; j+2<=cols*2; j+=2)
        {
            __m128d x0 = _mm_loadu_pd(row0+j);
            __m128d x1 = _mm_loadu_pd(row1+j);
            __m128d x2 = _mm_loadu_pd(row2+j);
            __m128d x3 = _mm_loadu_pd(row3+j);

            __m128d y0 = _mm_add_pd(x0, x1);
            __m128d y1 = _mm_sub_pd(x0, x1);
            __m128d y2 = _mm_add_pd(x2, x3);
#ifdef __SSE4_1__
            __m128d y3 = permute<_MM_SHUFFLE2(0,1)>(_mm_sub_pd(_mm_blend_pd(x2, x3, 0x1), _mm_blend_pd(x2, x3, 0x2)));
#else
            __m128d y3 = permute<_MM_SHUFFLE2(0,1)>(flip_even(_mm_sub_pd(x2, x3)));
#endif

            __m128d z0 = _mm_add_pd(y0, y2);
            __m128d z1 = _mm_add_pd(y1, y3);
            __m128d z2 = _mm_sub_pd(y0, y2);
            __m128d z3 = _mm_sub_pd(y1, y3);

            _mm_storeu_pd(row0+j, z0);
            _mm_storeu_pd(row1+j, z1);
            _mm_storeu_pd(row2+j, z2);
            _mm_storeu_pd(row3+j, z3);
        }
    }
    void inverse(double *data, int stride, int cols)
    {
        double *row0 = data;
        double *row1 = row0+stride;
        double *row2 = row1+stride;
        double *row3 = row2+stride;

        int j = 0;
#ifdef __AVX__
        for (; j+4<=cols*2; j+=4)
        {
            __m256d x0 = _mm256_loadu_pd(row0+j);
            __m256d x1 = _mm256_loadu_pd(row1+j);
            __m256d x2 = _mm256_loadu_pd(row2+j);
            __m256d x3 = _mm256_loadu_pd(row3+j);

            __m256d y0 = _mm256_add_pd(x0, x1);
            __m256d y1 = _mm256_sub_pd(x0, x1);
            __m256d y2 = _mm256_add_pd(x2, x3);
            __m256d x3223 = _mm256_sub_pd(_mm256_blend_pd(x2, x3, 0xa), _mm256_blend_pd(x2, x3, 0x5));
            __m256d y3 = _mm256_permute_pd(x3223, _MM_SHUFFLE4x2(0, 1, 0, 1));

            __m256d z0 = _mm256_add_pd(y0, y2);
            __m256d z1 = _mm256_add_pd(y1, y3);
            __m256d z2 = _mm256_sub_pd(y0, y2);
            __m256d z3 = _mm256_sub_pd(y1, y3);

            _mm256_storeu_pd(row0+j, z0);
            _mm256_storeu_pd(row1+j, z1);
            _mm256_storeu_pd(row2+j, z2);
            _mm256_storeu_pd(row3+j, z3);
        }
#endif

        for (; j+2<=cols*2; j+=2)
        {
            __m128d x0 = _mm_loadu_pd(row0+j);
            __m128d x1 = _mm_loadu_pd(row1+j);
            __m128d x2 = _mm_loadu_pd(row2+j);
            __m128d x3 = _mm_loadu_pd(row3+j);

            __m128d y0 = _mm_add_pd(x0, x1);
            __m128d y1 = _mm_sub_pd(x0, x1);
            __m128d y2 = _mm_add_pd(x2, x3);
#ifdef __SSE4_1__
            __m128d y3 = permute<_MM_SHUFFLE2(0,1)>(_mm_sub_pd(_mm_blend_pd(x2, x3, 0x2), _mm_blend_pd(x2, x3, 0x1)));
#else
            __m128d y3 = permute<_MM_SHUFFLE2(0,1)>(flip_even(_mm_sub_pd(x3, x2)));
#endif

            __m128d z0 = _mm_add_pd(y0, y2);
            __m128d z1 = _mm_add_pd(y1, y3);
            __m128d z2 = _mm_sub_pd(y0, y2);
            __m128d z3 = _mm_sub_pd(y1, y3);

            _mm_storeu_pd(row0+j, z0);
            _mm_storeu_pd(row1+j, z1);
            _mm_storeu_pd(row2+j, z2);
            _mm_storeu_pd(row3+j, z3);
        }
    }
};

template <>
struct FFTVertDouble<2>
{
    void forward(double *data, int stride, int cols)
    {
        double *row0 = data;
        double *row1 = row0+stride;

        int j = 0;
#ifdef __AVX__
        for (; j+8<=cols*2; j+=8)
        {
            __m256d x0 = _mm256_loadu_pd(row0+j);
            __m256d x1 = _mm256_loadu_pd(row1+j);
            _mm256_storeu_pd(row0+j, _mm256_add_pd(x0, x1));
            _mm256_storeu_pd(row1+j, _mm256_sub_pd(x0, x1));
        }
#endif
#ifdef __SSE2__
        for (; j+4<=cols*2; j+=4)
        {
            __m128d x0 = _mm_loadu_pd(row0+j);
            __m128d x1 = _mm_loadu_pd(row1+j);
            _mm_storeu_pd(row0+j, _mm_add_pd(x0, x1));
            _mm_storeu_pd(row1+j, _mm_sub_pd(x0, x1));
        }
#endif
        for (; j<cols*2; j++)
        {
            double x0 = row0[j];
            double x1 = row1[j];
            row0[j] = x0 + x1;
            row1[j] = x0 - x1;
        }
    }

    void inverse(double *data, int stride, int cols)
    {
        return forward(data, stride, cols);
    }
};

template <int N>
struct FFTVertImplSelector<N, double>
{
    typedef impl::FFTVertDouble<N> type;
};

} // impl
} // genfft

#endif

#endif /* GEN_FFT_DOUBLE_H */
