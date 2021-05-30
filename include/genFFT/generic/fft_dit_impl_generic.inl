/*
Copyright 2021 Michal Zientkiewicz

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

template <typename T>
void adjust_DIT_impl(std::complex<T> *F, const std::complex<T> *Z, int N, bool half, const T *twiddle) noexcept
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
    for (int i = 1, j = halfN-1; i < halfN/2; i++, j--)
    {
        std::complex<T> Ei = (Z[i] + conj(Z[j])) * T(0.5);
        std::complex<T> Oi = (Z[i] - conj(Z[j])) * T(0.5);
        std::complex<T> ti = { twiddle[2*i], twiddle[2*i+1] };
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

template <int N, typename T>
struct FFTDIT : impl::FFTDITBase<T>
{
    FFTDIT(int n) { Init(n); }
    void Init(int n)
    {
        assert(n == N || N < 0);
        this->n = n;
        twiddle  = genfft::DITTwiddle<N, T>(n);
    }

    void apply(T *F, const T *Z, bool half) const noexcept override
    {
        adjust_DIT_impl((std::complex<T>*)F, (const std::complex<T>*)Z, n, half, twiddle.t);
    }

private:
    int n = N > 0 ? N : 0;
    DITTwiddle<N, T> twiddle;

public:
    void *operator new(size_t count)
    {
        return aligned_alloc_raw(count, alignof(FFTDIT));
    }
    void operator delete(void *p)
    {
        free(p);
    }

    static std::shared_ptr<impl::FFTDITBase<T>> GetInstance(int n)
    {
        if (N < 0)
            return std::shared_ptr<impl::FFTDITBase<T>>(new FFTDIT(n));

        assert(n == N);

        // This object is immutable, we can use cached variant

        auto shptr = instance.lock();
        if (!shptr)
        {
            shptr.reset(new FFTDIT(n));
            instance = shptr;
        }
        return shptr;
    }
private:
    static std::weak_ptr<impl::FFTDITBase<T>> instance;
};

template <int N, class T>
std::weak_ptr<impl::FFTDITBase<T>> FFTDIT<N, T>::instance;

template <class T>
inline std::shared_ptr<impl::FFTDITBase<T>> GetDITImpl(int n, T)
{
    assert((n & 3) == 0 && "n must be divisible by 4");
    switch (n)
    {
        // Use cached variants for some powers of 2
#define SELECT_FFT_LEVEL(x) case (1<<x): return FFTDIT<(1<<x), T>::GetInstance(n);
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
        return FFTDIT<-1, T>::GetInstance(n);
    }
}
