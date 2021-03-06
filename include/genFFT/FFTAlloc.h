/*
Copyright (C) 2019 Michal Zientkiewicz

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

#ifndef GEN_FFT_ALLOC_H
#define GEN_FFT_ALLOC_H

#include <stdlib.h>
#include <type_traits>

namespace genfft {

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
align(T value, T alignment)
{
    return (value + alignment - 1)&-alignment;
}

inline void *aligned_alloc_raw(size_t bytes, size_t alignment = 32)
{
    bytes = align(bytes, alignment);
    return aligned_alloc(alignment, bytes);
}

template <typename T>
T *aligned_alloc_T(size_t N, size_t alignment = alignof(T))
{
    return static_cast<T*>(aligned_alloc_raw(N * sizeof(T), alignment));
}

} // genfft

#endif // GEN_FFT_ALLOC_H
