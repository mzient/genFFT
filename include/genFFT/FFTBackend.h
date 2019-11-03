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

#ifndef GENFFT_BACKEND_H
#define GENFFT_BACKEND_H

#ifdef GENFFT_DISPATCH_BACKEND
#ifdef GENFFT_NATIVE_BACKEND
#error genFFT Native and Dispatch mode cannot be used together
#endif
#else
#define GENFFT_NATIVE_BACKEND
#endif

#include "generic/fft_impl_generic.h"

#if defined(__x86_64__) || defined(__amd64__) || defined(__i386__)

    #ifdef GENFFT_DISPATCH_BACKEND
    #include "x86/fft_x86_dispatch.h"
    namespace genfft {
    namespace backend {
        using impl_x86_dispatch::GetImpl;
    } // backend
    } // genfft
    #else
    #include "x86/fft_x86_native.h"
    namespace genfft {
    namespace backend {
        using impl_native::GetImpl;
    } // backend
    } // genfft
    #endif

#else
    namespace genfft {
    namespace backend {
        using impl_generic::GetImpl;
    } // backend
    } // genfft
#endif

#endif
