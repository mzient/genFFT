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
SOFTWARE, EVEN IF ADVISED  THE uint32_t CH DAMAGE.
*/

#include <genFFT/x86/x86_features.h>
#include <cpuid.h>
#include <stdint.h>
#include <stdio.h>

namespace genfft {

static inline int
__get_cpuid_count (uint32_t leaf, uint32_t subleaf,
		   uint32_t *eax, uint32_t *ebx,
		   uint32_t *ecx, uint32_t *edx)
{
  uint32_t ext = leaf & 0x80000000;
  uint32_t maxlevel = __get_cpuid_max(ext, 0);

  if (maxlevel == 0 || maxlevel < leaf)
    return 0;

  __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
  return 1;
}

inline cpu_features InitCPUFeatures()
{
    cpu_features ret = {};
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    eax = 1;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx))
    {
        ret.AVX = ecx & bit_AVX;
        ret.SSE3 = edx & bit_SSE3;
        ret.SSSE3 = ecx & bit_SSSE3;
        ret.SSE41 = ecx & bit_SSE4_1;
        ret.SSE42 = ecx & bit_SSE4_2;
        ret.FMA = ecx & bit_FMA;

        ret.SSE = edx & bit_SSE;
        ret.SSE2 = edx & bit_SSE2;
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
    {
        ret.AVX2 = ebx & bit_AVX2;
    }
    return ret;
}

cpu_features x86_cpu_features = InitCPUFeatures();

cpu_features GetCPUFeatures()
{
    return x86_cpu_features;
}

} // genfft
