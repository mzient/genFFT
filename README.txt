genFFT - a simple template-based FFT with x86 backend.

The library's main goals are:
- usage simplicity
- implementation clarity
- good performance on x86 targets.
The library's performance depends heavily on compiler's optimizations and,
as such, should be used at least with -O2 flag or equivalent.

The library features x86 SIMD backends for both single- and double-precision numbers.
Currently it only supports power-of-two sizes.

The library can be used in one of 2 ways:
1. Native (header-only)
2. Dispatch (prebuilt)

Native variant is easy to integrate. It is the recommended variant for code running
on development machine or when the target architecture is known.

Dispatch mode builds the library with various CPU extensions on or off and the
actual implementation is chosen at run-time on the client machine. This variant
is useful for binary distribution.


Usage:
#include <genFFT/fft.h>

float out[2*1024], in[2*1024], inv[2*1024];
genfft::FFT<float>(1024);
fft.forward(out, in);
fft.inverse(inv, out); // no scaling!!!


Q&A:
Q: Why yet another FFT?
A: Out of need :) Existing libraries had one of the following problems:
    - license (GPL,non-commercial use, etc.)
    - no native support for 2D transform
    - no native support for single-precision data
    - slow
   genFFT addresses all of the above

Q: Does the library work with <compiler name here>
A: I don't know. It's only been tested with g++ 5x and clang 3.8; comments on
   github are welcome, I will try to fix compilation errors, should any appear
   on different compilers.

Q: Do you intend to add support for <enter architecture name here>
A: Not in foreseeable future.

Q: Do you plan to add half-spectrum inverse FFT for real data?
A: Maybe

Copyright (c) 2017-2019 Michal Zientkiewicz
See license.txt for details.
