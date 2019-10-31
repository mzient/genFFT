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

#ifndef _GENFFT_X86_H
#define _GENFFT_X86_H

#ifdef __SSE__
#include <xmmintrin.h>

#define GENFFT_USE_SSE
#endif

#ifdef __SSE3__
#define GENFFT_USE_SSE3
#include <emmintrin.h>
#endif

#ifdef __SSE4_1__
#include <smmintrin.h>

#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__FMA__)
#include <immintrin.h>
#endif

#ifdef __AVX__
#define GENFFT_USE_AVX
#endif

#ifdef __FMA__
#define GENFFT_USE_FMA
#endif

#ifdef __AVX2__
#define GENFFT_USE_AVX2
#endif

namespace genfft {
namespace impl_native {

#include "fft_float_impl_x86.inl"
#include "fft_double_impl_x86.inl"

}

} // genfft


#endif
