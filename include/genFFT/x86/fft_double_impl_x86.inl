/*
Copyright 2017-2020 Michal Zientkiewicz

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

// Single row FFT for single precision doubleing point numbers

template <int N>
struct FFTDouble
{
    FFTDouble<N/2> next;

    template <bool inv>
    void transform_impl(double *data)
    {
        next.template transform_impl<inv>(data);
        next.template transform_impl<inv>(data+N);

#ifdef GENFFT_USE_AVX
        for (int i=0; i<N; i+=4)
        {
            __m256d W = _mm256_load_pd(twiddle.t + i);
            __m256d E = _mm256_loadu_pd(data+i);
            __m256d O = _mm256_loadu_pd(data+i+N);
            __m256d Wr = _mm256_permute_pd(W, _MM_SHUFFLE4x2(0, 0, 0, 0));
            __m256d Wi = _mm256_permute_pd(W, _MM_SHUFFLE4x2(1, 1, 1, 1));

            __m256d OxWi = _mm256_mul_pd(O, Wi);
            __m256d Wxiperm = _mm256_permute_pd(OxWi, _MM_SHUFFLE4x2(0, 1, 0, 1));
#ifdef GENFFT_USE_FMA
            __m256d OxW  = inv ? _mm256_fmsubadd_pd(O, Wr, Wxiperm)
                               : _mm256_fmaddsub_pd(O, Wr, Wxiperm);
#else
            __m256d OxWr = _mm256_mul_pd(O, Wr);
            __m256d OxW  = _mm256_addsub_pd(OxWr, inv ? -Wxiperm : Wxiperm);
#endif
            __m256d lo = _mm256_add_pd(E, OxW);
            __m256d hi = _mm256_sub_pd(E, OxW);

            _mm256_storeu_pd(data + i,     lo);
            _mm256_storeu_pd(data + i + N, hi);
        }
#elif defined GENFFT_USE_SSE2
        for (int i=0; i<N; i+=2)
        {
            __m128d W = _mm_load_pd(twiddle.t + i);
            __m128d E = _mm_loadu_pd(data+i);
            __m128d O = _mm_loadu_pd(data+i+N);

            __m128d Wr = _mm_shuffle_pd(W, W, _MM_SHUFFLE2(0, 0));
            __m128d Wi = _mm_shuffle_pd(W, W, _MM_SHUFFLE2(1, 1));

            __m128d OxWi = _mm_mul_pd(O, Wi);
            __m128d OxWr = _mm_mul_pd(O, Wr);
            __m128d Wxiperm = permute<_MM_SHUFFLE2(0, 1)>(OxWi);
            __m128d OxW  = inv ? subadd(OxWr, Wxiperm) : addsub(OxWr, Wxiperm);
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
struct FFTDouble<1> : impl_generic::FFTGeneric<1, double> {};

template <>
struct FFTDouble<2> : impl_generic::FFTGeneric<2, double> {};

template <>
struct FFTDouble<4>
{
    template <bool inv>
    void transform_impl(double *data)
    {
        __m128d x45 = _mm_loadu_pd(data+4);
        __m128d x67 = _mm_loadu_pd(data+6);

        __m128d y45 = _mm_add_pd(x45, x67);

        __m128d y76 = inv ? _mm_sub_pd(x67, x45) : _mm_sub_pd(x45, x67);
        __m128d y67 = permute<_MM_SHUFFLE2(0,1)>(flip_even(y76));

        __m128d x01 = _mm_loadu_pd(data);
        __m128d x23 = _mm_loadu_pd(data+2);

        __m128d y01 = _mm_add_pd(x01, x23);
        __m128d y23 = _mm_sub_pd(x01, x23);
#ifdef GENFFT_USE_AVX
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

// Vertical multi-column FFT for singgle precision doubleing point values

template <int N>
struct FFTVertDouble
{
    FFTVertDouble<N/2> next;

    template <bool inv>
    void transform_impl(double *data, int stride, int cols)
    {
        const int half = N/2*stride;
        int next_col = cols;
        // Process the input in vertical spans, 32 complex numbers wide.
        // The last span may be wider, up to 48.
        for (int col=0; col<cols; col=next_col)
        {
            next_col = cols - col >= 48 ? col + 32 : cols;
            int span_width = next_col - col;
            double *span_data = data + 2*col;
            transform_span<inv>(span_data, stride, span_width);
        }
    }

    template <bool inv>
    void transform_span(double *data, int stride, int cols)
    {
        int half = N/2 * stride;
        next.template transform_impl<inv>(data,      stride, cols);
        next.template transform_impl<inv>(data+half, stride, cols);

        for (int i=0; i<N/2; i++)
        {
            double *even = data + i*stride;
            double *odd  = even + half;

            int j=0;
#ifdef GENFFT_USE_AVX
            __m256d Wr = _mm256_broadcast_sd(&twiddle.t[2*i]);
            __m256d Wi = _mm256_broadcast_sd(&twiddle.t[2*i+1]);

            for (; j+4<=2*cols; j+=4)
            {
                __m256d E = _mm256_loadu_pd(even+j);
                __m256d O = _mm256_loadu_pd(odd+j);

                __m256d OxWi = _mm256_mul_pd(O, Wi);
                __m256d OxWiperm = _mm256_permute_pd(OxWi, _MM_SHUFFLE4x2(0, 1, 0, 1));
#ifdef GENFFT_USE_FMA
                __m256d OxW  = inv ? _mm256_fmsubadd_pd(O, Wr, OxWiperm)
                                   : _mm256_fmaddsub_pd(O, Wr, OxWiperm);
#else
                __m256d OxWr = _mm256_mul_pd(O, Wr);
                __m256d OxW  = _mm256_addsub_pd(OxWr, inv ? -OxWiperm : OxWiperm);
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

            for (; j<2*cols; j+=2)
            {
                __m128d E = _mm_loadu_pd(even+j);
                __m128d O = _mm_loadu_pd(odd+j);

                __m128d OxWi = _mm_mul_pd(O, Wi128);
                __m128d Wxiperm = permute<_MM_SHUFFLE2(0, 1)>(OxWi);
#ifdef GENFFT_USE_FMA
                __m128d OxW  = inv ? _mm_fmsubadd_pd(O, Wr128, Wxiperm)
                                   : _mm_fmaddsub_pd(O, Wr128, Wxiperm);
#else
                __m128d OxWr = _mm_mul_pd(O, Wr128);
                __m128d OxW  = addsub(OxWr, inv ? -Wxiperm : Wxiperm);
#endif
                __m128d lo = _mm_add_pd(E, OxW);
                __m128d hi = _mm_sub_pd(E, OxW);

                _mm_storeu_pd(even+j, lo);
                _mm_storeu_pd(odd+j,  hi);
            }
        }
    }


    const Twiddle<N, double> twiddle;
};


template <>
struct FFTVertDouble<4>
{
    template <bool inv>
    void transform_impl(double *data, int stride, int cols)
    {
        double *row0 = data;
        double *row1 = row0+stride;
        double *row2 = row1+stride;
        double *row3 = row2+stride;

        int j = 0;
#ifdef GENFFT_USE_AVX
        for (; j+4<=cols*2; j+=4)
        {
            __m256d x0 = _mm256_loadu_pd(row0+j);
            __m256d x1 = _mm256_loadu_pd(row1+j);
            __m256d x2 = _mm256_loadu_pd(row2+j);
            __m256d x3 = _mm256_loadu_pd(row3+j);

            __m256d y0 = _mm256_add_pd(x0, x1);
            __m256d y1 = _mm256_sub_pd(x0, x1);
            __m256d y2 = _mm256_add_pd(x2, x3);
            enum {
                mask1 = inv ? 0xa : 0x5,
                mask2 = inv ? 0x5 : 0xa
            };
            __m256d x2332 = _mm256_sub_pd(_mm256_blend_pd(x2, x3, mask1), _mm256_blend_pd(x2, x3, mask2));
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
            enum {
                mask1 = inv ? 2 : 1,
                mask2 = inv ? 1 : 2
            };
#ifdef GENFFT_USE_SSE4_1
            __m128d y3 = permute<_MM_SHUFFLE2(0,1)>(_mm_sub_pd(_mm_blend_pd(x2, x3, mask1), _mm_blend_pd(x2, x3, mask2)));
#else
            __m128d y3 = permute<_MM_SHUFFLE2(0,1)>(flip_even(inv ? _mm_sub_pd(x3, x2) : _mm_sub_pd(x2, x3)));
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
    template <bool inv_unused>
    void transform_impl(double *data, int stride, int cols)
    {
        double *row0 = data;
        double *row1 = row0+stride;

        int j = 0;
#ifdef GENFFT_USE_AVX
        for (; j+4<=cols*2; j+=4)
        {
            __m256d x0 = _mm256_loadu_pd(row0+j);
            __m256d x1 = _mm256_loadu_pd(row1+j);
            _mm256_storeu_pd(row0+j, _mm256_add_pd(x0, x1));
            _mm256_storeu_pd(row1+j, _mm256_sub_pd(x0, x1));
        }
#endif
#ifdef GENFFT_USE_SSE2
        for (; j+2<=cols*2; j+=2)
        {
            __m128d x0 = _mm_loadu_pd(row0+j);
            __m128d x1 = _mm_loadu_pd(row1+j);
            _mm_storeu_pd(row0+j, _mm_add_pd(x0, x1));
            _mm_storeu_pd(row1+j, _mm_sub_pd(x0, x1));
        }
#else
        for (; j<cols*2; j++)
        {
            double x0 = row0[j];
            double x1 = row1[j];
            row0[j] = x0 + x1;
            row1[j] = x0 - x1;
        }
#endif
    }
};

template <>
struct FFTVertDouble<1>
{
    template <bool inv_unused>
    void transform_impl(double *data, int stride, int cols) {}
};

inline std::shared_ptr<impl::FFTBase<double>> GetImpl(int n, double)
{
    switch (n)
    {
#define SELECT_FFT_LEVEL(x) case (1<<x): return impl::FFTLevel<(1<<x), double, FFTDouble<(1<<x)>>::GetInstance();
            SELECT_FFT_LEVEL(0);
            SELECT_FFT_LEVEL(1);
            SELECT_FFT_LEVEL(2);
            SELECT_FFT_LEVEL(3);
            SELECT_FFT_LEVEL(4);
            SELECT_FFT_LEVEL(5);
            SELECT_FFT_LEVEL(6);
            SELECT_FFT_LEVEL(7);
            SELECT_FFT_LEVEL(8);
            SELECT_FFT_LEVEL(9);
            SELECT_FFT_LEVEL(10);
            SELECT_FFT_LEVEL(11);
            SELECT_FFT_LEVEL(12);
            SELECT_FFT_LEVEL(13);
            SELECT_FFT_LEVEL(14);
            SELECT_FFT_LEVEL(15);
            SELECT_FFT_LEVEL(16);
            SELECT_FFT_LEVEL(17);
            SELECT_FFT_LEVEL(18);
            SELECT_FFT_LEVEL(19);
            SELECT_FFT_LEVEL(20);
            SELECT_FFT_LEVEL(21);
            SELECT_FFT_LEVEL(22);
            SELECT_FFT_LEVEL(23);
#undef SELECT_FFT_LEVEL
        default:
            assert(!"unsupported size");
    }
}

inline std::shared_ptr<impl::FFTVertBase<double>> GetVertImpl(int n, double)
{
    switch (n)
    {
#define SELECT_FFT_LEVEL(x) case (1<<x): return impl::FFTVertLevel<(1<<x), double, FFTVertDouble<(1<<x)>>::GetInstance();
            SELECT_FFT_LEVEL(0);
            SELECT_FFT_LEVEL(1);
            SELECT_FFT_LEVEL(2);
            SELECT_FFT_LEVEL(3);
            SELECT_FFT_LEVEL(4);
            SELECT_FFT_LEVEL(5);
            SELECT_FFT_LEVEL(6);
            SELECT_FFT_LEVEL(7);
            SELECT_FFT_LEVEL(8);
            SELECT_FFT_LEVEL(9);
            SELECT_FFT_LEVEL(10);
            SELECT_FFT_LEVEL(11);
            SELECT_FFT_LEVEL(12);
            SELECT_FFT_LEVEL(13);
            SELECT_FFT_LEVEL(14);
            SELECT_FFT_LEVEL(15);
            SELECT_FFT_LEVEL(16);
            SELECT_FFT_LEVEL(17);
            SELECT_FFT_LEVEL(18);
            SELECT_FFT_LEVEL(19);
            SELECT_FFT_LEVEL(20);
            SELECT_FFT_LEVEL(21);
            SELECT_FFT_LEVEL(22);
            SELECT_FFT_LEVEL(23);
#undef SELECT_FFT_LEVEL
        default:
            assert(!"unsupported size");
    }
}
