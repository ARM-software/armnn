//
// Copyright © 2020 Arm Ltd. All rights reserved.
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
