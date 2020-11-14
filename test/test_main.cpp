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

#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>

#if defined(__x86_64__) || defined(__i386__)
#include <genFFT/x86/x86_features.h>

void PrintCPUFeatures()
{
    genfft::cpu_features features = genfft::GetCPUFeatures();
    #define PRINT_FEATURE(name)\
        std::cout << std::left << std::setw(10) << #name ":" << (features.name ? "yes" : "no") << "\n";
    PRINT_FEATURE(SSE)
    PRINT_FEATURE(SSE2)
    PRINT_FEATURE(SSE3)
    PRINT_FEATURE(SSSE3)
    PRINT_FEATURE(SSE41)
    PRINT_FEATURE(SSE42)
    PRINT_FEATURE(AVX)
    PRINT_FEATURE(AVX2)
    PRINT_FEATURE(FMA)
    std::cout << std::flush;
}
#else
void PrintCPUFeatures()
{
}
#endif

int main(int argc, char *argv[])
{
    PrintCPUFeatures();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
