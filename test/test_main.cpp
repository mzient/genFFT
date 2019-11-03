#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>

#if defined(__x86_64__) || defined(__i386__)
#include "../src/x86_features.h"

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
