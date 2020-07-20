//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

// This header brings in the Win32 API header, with some small modifications applied to prevent clashes with our code.

#if defined(_MSC_VER)

#define NOMINMAX    // Prevent definition of min/max macros that interfere with std::min/max
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
// Windows.h defines some names that we don't need and interfere with some of our definition
#undef TIME_MS      // Instrument.hpp
#undef CreateEvent  // ITimelineDecoder.hpp

#endif

#if defined(__MINGW32__)

#define NOMINMAX    // Prevent definition of min/max macros that interfere with std::min/max
#define WIN32_LEAN_AND_MEAN
#define WINVER 0x0A00
#define _WIN32_WINNT 0x0A00
#include <windows.h>
// Windows.h defines some names that we don't need and interfere with some of our definition
#undef TIME_MS      // Instrument.hpp
#undef CreateEvent  // ITimelineDecoder.hpp

#endif
